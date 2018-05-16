#include <cuda.h>
#include <pthread.h>
#include <rte_atomic.h>

#include "onvm_init.h"
#include "onvm_common.h"
#include "manager.h"
#include "drvapi_error_string.h"

extern struct onvm_service_chain *default_chain;

static pthread_mutex_t lock;

struct rte_mempool *nf_request_pool, *nf_response_pool;
struct rte_ring *nf_request_queue;

static rte_atomic32_t allocated_sm;
static CUcontext context;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
static inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
	if (CUDA_SUCCESS != err) {
		fprintf(stderr, "CUDA Driver API error = %04d  \"%s\" from file <%s>, line %i.\n",
				err, getCudaDrvErrorString(err), file, line );
		exit(-1);
	}
}

void
init_manager(void)
{
	int device_count = 0;
	CUdevice device;

	CUresult err = cuInit(0);
	if (err == CUDA_SUCCESS) {
		checkCudaErrors(cuDeviceGetCount(&device_count));
		if (device_count == 0) {
			fprintf(stderr, "Error: no devices supporting CUDA\n");
			exit(-1);
		}
	}

	// get first CUDA device
	checkCudaErrors(cuDeviceGet(&device, 0));
	char name[100];
	cuDeviceGetName(name, 100, device);
	printf("> Using device 0: %s\n", name);

	// get compute capabilities and the devicename
	int major = 0, minor = 0;
	checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
	printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

	size_t totalGlobalMem;
	checkCudaErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
	printf("  Total amount of global memory:   %llu bytes\n",
			(unsigned long long)totalGlobalMem);
	printf("  64-bit Memory Address:           %s\n",
			(totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
			"YES" : "NO");

	checkCudaErrors( cuCtxCreate(&context, 0, device) );

	if (pthread_mutex_init(&lock, NULL) != 0)
		rte_exit(EXIT_FAILURE, "Cannot init lock\n");


	int i, j;
	const struct rte_memzone *mz_gpu;

	/* Create a single queue for NFs to send requests to scheduler */
	nf_request_queue = rte_ring_create(
			_NF_REQUEST_QUEUE_NAME,
			MAX_CLIENTS * 16,
			rte_socket_id(),
			RING_F_SC_DEQ); // MP enqueue (default), SC dequeue

	if (nf_request_queue == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create nf request queue\n");

	/* Create multiple queues for scheduler to send responses to NFs */
	for (i = 0; i < MAX_CLIENTS; i++) {
		/* Create an response queue and a stream for each thread */
		assert(clients[i].instance_id == i);

		for (j = 0; j < MAX_CONCURRENCY_NUM; j ++) {
			clients[i].response_q[j] = rte_ring_create(
					get_rsp_queue_name(i, j),
					CLIENT_QUEUE_RINGSIZE, rte_socket_id(),
					RING_F_SP_ENQ | RING_F_SC_DEQ); /* single producer, single consumer */
			if (clients[i].response_q[j] == NULL)
				rte_exit(EXIT_FAILURE, "Cannot create response ring queue for client %d\n", i);

			checkCudaErrors( cuStreamCreate(&(clients[i].stream[j]), CU_STREAM_NON_BLOCKING) );
		}

		clients[i].global_response_q = rte_ring_create(
				get_rsp_queue_name(i, GLOBAL_RSP_QUEUE),
				CLIENT_QUEUE_RINGSIZE, rte_socket_id(),
				RING_F_SP_ENQ | RING_F_SC_DEQ); /* single producer, single consumer */
		if (clients[i].global_response_q == NULL)
			rte_exit(EXIT_FAILURE, "Cannot create response ring queue for global response queue\n");

		/* Set up the shared GPU scheduling info area */
		mz_gpu = rte_memzone_reserve(
				get_gpu_info_name(i),
				sizeof(struct gpu_schedule_info),
				rte_socket_id(), NO_FLAGS);
		if (mz_gpu == NULL)
			rte_exit(EXIT_FAILURE, "Cannot reserve memory zone for client information\n");
		memset(mz_gpu->addr, 0, sizeof(struct gpu_schedule_info));

		clients[i].gpu_info = mz_gpu->addr;
	}

	nf_request_pool = rte_mempool_create(_NF_REQUEST_MEMPOOL_NAME, MAX_REQUEST_NUM,
			sizeof(struct nf_req), 0, 0, NULL, NULL, NULL, NULL, rte_socket_id(), NO_FLAGS);
	if (nf_request_pool == NULL)
		rte_exit(EXIT_FAILURE, "Fail to create nf_request_pool\n");

	nf_response_pool = rte_mempool_create(_NF_RESPONSE_MEMPOOL_NAME, MAX_RESPONSE_NUM,
			sizeof(struct nf_rsp), 0, 0, NULL, NULL, NULL, NULL, rte_socket_id(), NO_FLAGS);
	if (nf_response_pool == NULL)
		rte_exit(EXIT_FAILURE, "Fail to create nf_response_pool\n");
	
	rte_atomic32_set(&allocated_sm, 0);
}

void
manager_nf_init(int instance_id)
{
	struct client *cl = &(clients[instance_id]);
	if (cl->init == 1)
		rte_exit(EXIT_FAILURE, "Network function %d has been inited\n", instance_id);

	if (cl->gpu_info->init != 1)
		rte_exit(EXIT_FAILURE, "GPU INFO for this client has not been filled\n");

#if !defined(PKTGEN_FRAMEWORK)
	/* load GPU kernel */
	RTE_LOG(INFO, APP, "Loading CUDA, module_file: %s, function: %s\n", cl->gpu_info->module_file, cl->gpu_info->kernel_name);
	checkCudaErrors( cuModuleLoad(&(cl->module), cl->gpu_info->module_file) );
	checkCudaErrors( cuModuleGetFunction(&(cl->function), cl->module, cl->gpu_info->kernel_name) );
#endif

	/* Default values, waiting to be changed by the Scheduler.
	 * This works correctly, as the client would not use the batch size when it equals to 0 */
	cl->threads_per_blk = 1024;
	cl->blk_num = PRESET_BLK_NUM;
	cl->batch_size = PRESET_BATCH_SIZE;
	
	/* mark that this NF has been initiated */
	cl->init = 1;

	return;
}

static void
sync_stream_callback(CUstream cuda_stream, CUresult status, void *user_data)
{
	UNUSED(cuda_stream);
	checkCudaErrors(status);

	struct nf_rsp *rsp;
	struct nf_req *req = (struct nf_req *)user_data;
	struct client *cl = &(clients[req->instance_id]);

	if (rte_mempool_get(nf_response_pool, (void **)&rsp) < 0)
		rte_exit(EXIT_FAILURE, "Failed to get response memory\n");
	if (rsp == NULL)
		rte_exit(EXIT_FAILURE, "Response memory not allocated\n");

	rsp->type = RSP_GPU_SYNC_STREAM;
	rsp->stream_id = req->stream_id;
	rsp->states = RSP_SUCCESS;

	if (rte_ring_enqueue(cl->response_q[req->thread_id], rsp) < 0) {
		rte_mempool_put(nf_response_pool, rsp);
		rte_exit(EXIT_FAILURE, "Cannot enqueue into global response queue");
	}

	rte_mempool_put(nf_request_pool, req);
}

static void
release_sm(CUstream cuda_stream, CUresult status, void *user_data)
{
	UNUSED(cuda_stream);
	checkCudaErrors(status);
	int blk_num = (int)(intptr_t)user_data;
	rte_atomic32_sub(&allocated_sm, blk_num);
}

int
manager_thread_main(void *arg)
{
	UNUSED(arg);
	struct client *cl;
	struct nf_req *req;
	struct nf_rsp *rsp;
	const struct rte_memzone *mz;

	int blk_num;
	int threads_per_blk;
	int sid;

	CUdeviceptr dptr;
	void *host_ptr;
	uint64_t i, offset;

	checkCudaErrors( cuCtxSetCurrent(context) );

	RTE_LOG(INFO, APP, "Core %d: Manager is running\n", rte_lcore_id());

	while (1) {
		req = NULL;
		while (rte_ring_dequeue(nf_request_queue, (void **)&req) != 0) ;
		if (req == NULL)
			rte_exit(EXIT_FAILURE, "Dequeued request is NULL");

		switch(req->type) {
			case REQ_HOST_MALLOC:
				/* allocate from shared memory, each thread has one chance to malloc
				 * a large area of memory */
				mz = rte_memzone_reserve(
						get_buf_name(req->instance_id), 
						req->size,
						rte_socket_id(),
						NO_FLAGS);
				if (mz == NULL)
					rte_exit(EXIT_FAILURE, "Cannot reserve memory zone for client information\n");

				RTE_LOG(DEBUG, APP, "Host Alloc: %p, size %d\n", mz->addr, req->size);

				/* get response */
				if (rte_mempool_get(nf_response_pool, (void **)&rsp) < 0)
					rte_exit(EXIT_FAILURE, "Failed to get response memory\n");
				if (rsp == NULL)
					rte_exit(EXIT_FAILURE, "Response memory not allocated\n");

				rsp->type = RSP_HOST_MALLOC;
				rsp->states = RSP_SUCCESS;
				rsp->instance_id = req->instance_id;

				/* Enqueue and tell */
				cl = &(clients[req->instance_id]);
				if (rte_ring_enqueue(cl->response_q[req->thread_id], rsp) < 0) {
					rte_mempool_put(nf_response_pool, rsp);
					rte_exit(EXIT_FAILURE, "Cannot enqueue into response queue, thread %d\n", req->thread_id);
				}

				rte_mempool_put(nf_request_pool, req);
				break;

			case REQ_GPU_MALLOC:
				checkCudaErrors( cuMemAlloc(&dptr, req->size) );

				RTE_LOG(DEBUG, APP, "cuMemAlloc: %lx, size %d\n", (uint64_t)dptr, req->size);

				/* get response */
				if (rte_mempool_get(nf_response_pool, (void **)&rsp) < 0)
					rte_exit(EXIT_FAILURE, "Failed to get response memory\n");
				if (rsp == NULL)
					rte_exit(EXIT_FAILURE, "Response memory not allocated\n");

				rsp->type = RSP_GPU_MALLOC;
				rsp->states = RSP_SUCCESS;
				rsp->dev_ptr = dptr;

				cl = &(clients[req->instance_id]);
				if (rte_ring_enqueue(cl->response_q[req->thread_id], rsp) < 0) {
					rte_mempool_put(nf_response_pool, rsp);
					rte_exit(EXIT_FAILURE, "Cannot enqueue into response queue, thread %d\n", req->thread_id);
				}

				rte_mempool_put(nf_request_pool, req);
				break;

			case REQ_GPU_MEMCPY_HTOD_ASYNC:
				cl = &(clients[req->instance_id]);
				mz = rte_memzone_lookup(get_buf_name(req->instance_id));
				host_ptr = (char *)(mz->addr) + req->host_offset;
				rte_spinlock_lock(&cl->stats.update_lock);
				cl->stats.htod_mem += req->size;
				rte_spinlock_unlock(&cl->stats.update_lock);

				RTE_LOG(INFO, APP, "htod %d\n", req->stream_id);
				checkCudaErrors( cuMemcpyHtoDAsync(req->device_ptr, host_ptr, req->size, cl->stream[req->stream_id]) );

				rte_mempool_put(nf_request_pool, req);

				RTE_LOG(DEBUG, APP, "cuMemcpyHtoDAsync: %lx <- %p (%d), stream_id = %d\n", (uint64_t)req->device_ptr, host_ptr, req->size, req->stream_id);
				break;

			case REQ_GPU_MEMCPY_DTOH_ASYNC:
				cl = &(clients[req->instance_id]);
				mz = rte_memzone_lookup(get_buf_name(req->instance_id));
				host_ptr = (char *)(mz->addr) + req->host_offset;
				rte_spinlock_lock(&cl->stats.update_lock);
				cl->stats.dtoh_mem += req->size;
				rte_spinlock_unlock(&cl->stats.update_lock);

				checkCudaErrors( cuMemcpyDtoHAsync(host_ptr, req->device_ptr, req->size, cl->stream[req->stream_id]) );

				rte_mempool_put(nf_request_pool, req);

				RTE_LOG(DEBUG, APP, "cuMemcpyDtoHAsync: %p <- %lx (%d), stream_id = %d\n", host_ptr, (uint64_t)req->device_ptr, req->size, req->stream_id);
				break;

			case REQ_GPU_LAUNCH_STREAM_ASYNC:
				cl = &(clients[req->instance_id]);
				sid = req->stream_id;

				blk_num = cl->blk_num;
				threads_per_blk = cl->threads_per_blk;
				if (blk_num <= 0 || threads_per_blk < 128 || threads_per_blk > 1024) {
					rte_exit(EXIT_FAILURE, "instance id %d, blk_num %d, threads_per_blk %d\n", cl->instance_id, blk_num, threads_per_blk);
				}

				if (rte_atomic32_add_return(&allocated_sm, blk_num) > SM_TOTAL_NUM) {
					rte_exit(EXIT_FAILURE, "There should always have available SMs\n");
				}

				char *args = cl->gpu_info->args[sid];
				uint64_t *arg_info = (uint64_t *)(cl->gpu_info->arg_info[sid]);
				uint64_t arg_num = arg_info[0];

				for (i = 0; i < arg_num; i ++) {
					offset = arg_info[1 + i];
					arg_info[1 + i] = (uint64_t)((uint8_t *)args + offset);
				}

				checkCudaErrors( cuLaunchKernel(cl->function, 
							blk_num, 1, 1,  // Nx1x1 blocks
							threads_per_blk, 1, 1, // Mx1x1 threads
							0, cl->stream[sid], (void **)&(arg_info[1]), 0) );

				// will this be slow?
				checkCudaErrors( cuStreamAddCallback(cl->stream[req->stream_id], release_sm, (void *)(intptr_t)blk_num, 0) );

				rte_mempool_put(nf_request_pool, req);
				break;

			case REQ_GPU_SYNC_STREAM:
				cl = &(clients[req->instance_id]);

				checkCudaErrors( cuStreamAddCallback(cl->stream[req->stream_id], sync_stream_callback, (void *)req, 0) );
				break;

			case REQ_GPU_MEMFREE:
				checkCudaErrors( cuMemFree(req->device_ptr) );
				rte_mempool_put(nf_request_pool, req);
				break;

			case REQ_GPU_MEMSET:
				cl = &(clients[req->instance_id]);
				checkCudaErrors( cuMemsetD32Async(req->device_ptr, req->value, req->size, cl->stream[req->thread_id]) );
				rte_mempool_put(nf_request_pool, req);
				break;

			default:
				rte_exit(EXIT_FAILURE, "Wrong request type");
				rte_mempool_put(nf_request_pool, req);

				break;
		}
	}	

	return 0;
}
