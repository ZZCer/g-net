#include <cuda.h>
#include <pthread.h>

#include "onvm_init.h"
#include "onvm_common.h"
#include "manager.h"
#include "drvapi_error_string.h"

extern struct onvm_service_chain *default_chain;

CUdeviceptr gpu_pkts_buf, gpu_pkts_head;
rte_spinlock_t gpu_pkts_lock;


static pthread_mutex_t lock;
static struct nf_req *pending_req_header = NULL, *pending_req_tail = NULL;

struct rte_mempool *nf_request_pool, *nf_response_pool;
struct rte_ring *nf_request_queue;

static int allocated_sm = 0;
CUcontext context;

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
	checkCudaErrors( cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );
	checkCudaErrors( cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) );
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

		for (j = 0; j < MAX_CPU_THREAD_NUM; j ++) {
			clients[i].response_q[j] = rte_ring_create(
					get_rsp_queue_name(i, j),
					CLIENT_QUEUE_RINGSIZE, rte_socket_id(),
					RING_F_SP_ENQ | RING_F_SC_DEQ); /* single producer, single consumer */
			if (clients[i].response_q[j] == NULL)
				rte_exit(EXIT_FAILURE, "Cannot create response ring queue for client %d\n", i);

			checkCudaErrors( cuStreamCreate(&(clients[i].stream[j]), CU_STREAM_NON_BLOCKING) );
			checkCudaErrors( cuEventCreate(&(clients[i].kern_start[j]), CU_EVENT_DEFAULT) );
			checkCudaErrors( cuEventCreate(&(clients[i].kern_end[j]), CU_EVENT_DEFAULT) );
			checkCudaErrors( cuEventCreate(&(clients[i].gpu_start[j]), CU_EVENT_DEFAULT) );
			checkCudaErrors( cuEventCreate(&(clients[i].gpu_end[j]), CU_EVENT_DEFAULT) );
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

	rte_ring_create(RX_GPU_QUEUE, MAX_BATCH_SIZE * BATCH_QUEUE_FACTOR, rte_socket_id(), NO_FLAGS);
	checkCudaErrors( cuMemAlloc(&gpu_pkts_buf, GPU_BUF_SIZE * GPU_PKT_LEN) );
	gpu_pkts_head = gpu_pkts_buf;
	rte_spinlock_init(&gpu_pkts_lock);
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
sync_callback(CUstream cuda_stream, CUresult status, void *user_data)
{
	checkCudaErrors(status);

	struct nf_rsp *rsp;
	int instance_id = (int)(uint64_t)user_data;
	struct client *cl = &(clients[instance_id]);

	int i, thread_id = -1, allset = 1, res;

	for (i = 0; i < cl->worker_thread_num; i ++) {
		if (cuda_stream == cl->stream[i]) {
			thread_id = i;
			cl->sync[i] = 1;
		} else if (cl->sync[i] != 1) {
			/* There is still stream unfinished */
			allset = 0;
		}
	}

	if (thread_id == -1)
		rte_exit(EXIT_FAILURE, "Failed to find the cuda stream\n");

	/* release SM resource */
	/* If all streams are ended, send response */
	if (allset == 1) {
		/* allocate a response  */
		if (rte_mempool_get(nf_response_pool, (void **)&rsp) < 0)
			rte_exit(EXIT_FAILURE, "Failed to get response memory\n");
		if (rsp == NULL)
			rte_exit(EXIT_FAILURE, "Response memory not allocated\n");

		rsp->type = RSP_GPU_GLOBAL_SYNC;
		rsp->batch_size = cl->batch_size;

		res = rte_ring_enqueue(cl->global_response_q, rsp);
		if (res < 0) {
			rte_mempool_put(nf_response_pool, rsp);
			rte_exit(EXIT_FAILURE, "Cannot enqueue into global response queue");
		}

		RTE_LOG(DEBUG, APP, "Global Synchronized\n");
	}
}

static void
stream_sync_callback(CUstream cuda_stream, CUresult status, void *user_data)
{
	UNUSED(cuda_stream);
	checkCudaErrors(status);
	struct nf_req *req = (struct nf_req *)user_data;
	struct client *cl = &(clients[req->instance_id]);
	int tid = req->thread_id;

	checkCudaErrors( cuEventQuery(cl->kern_start[tid]) );
	checkCudaErrors( cuEventQuery(cl->kern_end[tid]) );
	checkCudaErrors( cuEventQuery(cl->gpu_start[tid]) );
	checkCudaErrors( cuEventQuery(cl->gpu_end[tid]) );

	float diff_ms;
	checkCudaErrors( cuEventElapsedTime(&diff_ms, cl->kern_start[tid], cl->kern_end[tid]) );
	cl->stats.kernel_time += diff_ms * 1000.0;
	checkCudaErrors( cuEventElapsedTime(&diff_ms, cl->gpu_start[tid], cl->gpu_end[tid]) );
	cl->stats.gpu_time += diff_ms * 1000.0;
	cl->stats.kernel_cnt ++;

	struct nf_rsp *rsp;
	if (rte_mempool_get(nf_response_pool, (void **)&rsp) < 0)
		rte_exit(EXIT_FAILURE, "Failed to get response memory\n");
	if (rsp == NULL)
		rte_exit(EXIT_FAILURE, "Response memory not allocated\n");

	rsp->type = RSP_GPU_SYNC_STREAM;

	int res = rte_ring_enqueue(cl->response_q[tid], rsp);
	if (res < 0) {
		rte_mempool_put(nf_response_pool, rsp);
		rte_exit(EXIT_FAILURE, "Cannot enqueue into global response queue");
	}

	rte_mempool_put(nf_request_pool, req);
}

static void
memcpy_callback(CUstream cuda_stream, CUresult status, void *user_data)
{
	UNUSED(cuda_stream);
	checkCudaErrors(status);

	struct nf_rsp *rsp;
	struct nf_req *req = (struct nf_req *)user_data;
	struct client *cl = &(clients[req->instance_id]);

#if defined(GRAPH_TIME)
	struct timespec tt;
	clock_gettime(CLOCK_MONOTONIC, &tt);
	if (req->type == REQ_GPU_MEMCPY_HTOD_SYNC) {
		printf("%d\t2\t%d\t%.2lf\n", req->instance_id, req->size, (double)1000000*tt.tv_sec + tt.tv_nsec/1000);
		graph_time_htod = 1;
	} else if (req->type == REQ_GPU_MEMCPY_DTOH_SYNC) {
		printf("%d\t6\t%d\t%.2lf\n", req->instance_id, req->size, (double)1000000*tt.tv_sec + tt.tv_nsec/1000);
		graph_time_dtoh = 1;
	} else {
		rte_exit(EXIT_FAILURE, "Wrong request type\n");
	}
#endif

	if (rte_mempool_get(nf_response_pool, (void **)&rsp) < 0)
		rte_exit(EXIT_FAILURE, "Failed to get response memory\n");
	if (rsp == NULL)
		rte_exit(EXIT_FAILURE, "Response memory not allocated\n");

	if (req->type == REQ_GPU_MEMCPY_HTOD_SYNC) {
		rsp->type = RSP_GPU_MEMCPY_HTOD_SYNC;
	} else if (req->type == REQ_GPU_MEMCPY_DTOH_SYNC) {
		rsp->type = RSP_GPU_MEMCPY_DTOH_SYNC;
	} else {
		rte_exit(EXIT_FAILURE, "Wrong type in memcpy callback");
	}
	rsp->states = RSP_SUCCESS;

	if (rte_ring_enqueue(cl->response_q[req->thread_id], rsp) < 0) {
		rte_mempool_put(nf_response_pool, rsp);
		rte_exit(EXIT_FAILURE, "Cannot enqueue into response queue, thread %d\n", req->thread_id);
	}

	/* Release the request */
	rte_mempool_put(nf_request_pool, req);
}

#if 0
static void
execute_pending_kernel(void)
{
	struct nf_req *req_ptr, *req;
	struct client *cl;
	uint64_t i, offset;
	int blk_num, threads_per_blk;

	/* Execute the pending kernels */
	for (req_ptr = pending_req_header; req_ptr != NULL; req_ptr = req_ptr->next) {
		req = req_ptr;
		cl = &(clients[req->instance_id]);
		blk_num = cl->blk_num;
		threads_per_blk = cl->threads_per_blk;

		if (req->type == REQ_GPU_LAUNCH) {
			/* Allocate resource to all the threads(streams) of the NF at one time */
			rte_exit(EXIT_FAILURE, "REQ_GPU_LAUNCH not supported");

		} else if (req->type == REQ_GPU_LAUNCH_ALL_STREAM) {
			pthread_mutex_lock(&lock);
			if (cl->blk_num * cl->worker_thread_num > SM_TOTAL_NUM - allocated_sm) {
				pthread_mutex_unlock(&lock);
				break;
			}
			/* Enough SMs left */
			allocated_sm += blk_num * cl->worker_thread_num;
			pthread_mutex_unlock(&lock);

			int tid;
			char *args;
			uint64_t *arg_info;
			uint64_t arg_num;

			for (tid = 0; tid < cl->worker_thread_num; tid ++) {
				args = cl->gpu_info->args[tid];
				arg_info = (uint64_t *)(cl->gpu_info->arg_info[tid]);
				arg_num = arg_info[0];

				/* Transfer all argument offsets into pointers */
				for (i = 0; i < arg_num; i ++) {
					offset = arg_info[1 + i];
					arg_info[1 + i] = (uint64_t)((uint8_t *)args + offset);
				}

				type_arg a;
				a.info.instance_id = cl->instance_id;
				a.info.blk_num = blk_num;

				/* Launch Kernel */
				checkCudaErrors( cuLaunchKernel(cl->function, 
							blk_num, 1, 1,  // Nx1x1 blocks
							threads_per_blk, 1, 1, // 1x1x1 threads
							0, cl->stream[tid], (void **)&(arg_info[1]), 0) );

				cl->sync[tid] = 0;
				checkCudaErrors( cuStreamAddCallback(cl->stream[tid], stream_callback, (void *)a.arg, 0) );

				/* Release the request */
				rte_mempool_put(nf_request_pool, req);
			}

			/* Update the pending rsp pointers */
			pending_req_header = req_ptr->next;
			if (pending_req_header == NULL) {
				/* Launched all kernels, no rsp left in the list */
				pending_req_tail = NULL;
			}

		}
	}
}
#endif

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
	int tid;

	CUdeviceptr dptr;
	void *host_ptr;
	uint64_t i, offset;
#if !defined(SYNC_MODE)
	unsigned int record_blk_num[MAX_CLIENTS];
	unsigned int record_blk_num_thread[MAX_CLIENTS][MAX_CPU_THREAD_NUM];
#endif

	checkCudaErrors( cuCtxSetCurrent(context) );

	RTE_LOG(INFO, APP, "Core %d: Manager is running\n", rte_lcore_id());

	while (1) {
		/* Try to execute pending kernels first */
		//execute_pending_kernel();

		// each NF send req in a batch, from the first memcpyHtoD to the last memcpyDtoH
		req = NULL;
		while (rte_ring_dequeue(nf_request_queue, (void **)&req) != 0) ;
		if (req == NULL)
			rte_exit(EXIT_FAILURE, "Dequeued request is NULL");

		switch(req->type) {
			case REQ_HOST_MALLOC:
				/* allocate from shared memory, each thread has one chance to malloc
				 * a large area of memory */
				mz = rte_memzone_reserve(
						get_buf_name(req->instance_id, req->thread_id), 
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
				mz = rte_memzone_lookup(get_buf_name(req->instance_id, req->thread_id));
				host_ptr = (char *)(mz->addr) + req->host_offset;
				rte_spinlock_lock(&cl->stats.update_lock);
				cl->stats.htod_mem += req->size;
				rte_spinlock_unlock(&cl->stats.update_lock);

				checkCudaErrors( cuMemcpyHtoDAsync(req->device_ptr, host_ptr, req->size, cl->stream[req->thread_id]) );

				rte_mempool_put(nf_request_pool, req);

				RTE_LOG(DEBUG, APP, "cuMemcpyHtoDAsync: %lx <- %p (%d), thread_id = %d\n", (uint64_t)req->device_ptr, host_ptr, req->size, req->thread_id);
				break;

			case REQ_GPU_MEMCPY_HTOD_SYNC:
				cl = &(clients[req->instance_id]);
				mz = rte_memzone_lookup(get_buf_name(req->instance_id, req->thread_id));
				host_ptr = (char *)(mz->addr) + req->host_offset;
				rte_spinlock_lock(&cl->stats.update_lock);
				cl->stats.htod_mem += req->size;
				rte_spinlock_unlock(&cl->stats.update_lock);

				checkCudaErrors( cuMemcpyHtoDAsync(req->device_ptr, host_ptr, req->size, cl->stream[req->thread_id]) );

				checkCudaErrors( cuStreamAddCallback(cl->stream[req->thread_id], memcpy_callback, (void *)req, 0) );

				RTE_LOG(DEBUG, APP, "cuMemcpyHtoDAsync[SYNC]: %lx <- %p (%d), thread_id = %d\n", (uint64_t)req->device_ptr, host_ptr, req->size, req->thread_id);
				break;

			case REQ_GPU_MEMCPY_DTOH_ASYNC:
				cl = &(clients[req->instance_id]);
				mz = rte_memzone_lookup(get_buf_name(req->instance_id, req->thread_id));
				host_ptr = (char *)(mz->addr) + req->host_offset;
				rte_spinlock_lock(&cl->stats.update_lock);
				cl->stats.dtoh_mem += req->size;
				rte_spinlock_unlock(&cl->stats.update_lock);

				checkCudaErrors( cuMemcpyDtoHAsync(host_ptr, req->device_ptr, req->size, cl->stream[req->thread_id]) );

				rte_mempool_put(nf_request_pool, req);

				RTE_LOG(DEBUG, APP, "cuMemcpyDtoHAsync: %p <- %lx (%d), thread_id = %d\n", host_ptr, (uint64_t)req->device_ptr, req->size, req->thread_id);
				break;

			case REQ_GPU_MEMCPY_DTOH_SYNC:
				cl = &(clients[req->instance_id]);
				mz = rte_memzone_lookup(get_buf_name(req->instance_id, req->thread_id));
				host_ptr = (char *)(mz->addr) + req->host_offset;
				rte_spinlock_lock(&cl->stats.update_lock);
				cl->stats.dtoh_mem += req->size;
				rte_spinlock_unlock(&cl->stats.update_lock);

				checkCudaErrors( cuMemcpyDtoHAsync(host_ptr, req->device_ptr, req->size, cl->stream[req->thread_id]) );

				checkCudaErrors( cuStreamAddCallback(cl->stream[req->thread_id], memcpy_callback, (void *)req, 0) );

				RTE_LOG(DEBUG, APP, "cuMemcpyDtoHAsync[SYNC]: %p <- %lx (%d), thread_id = %d\n", host_ptr, (uint64_t)req->device_ptr, req->size, req->thread_id);
				break;

			case REQ_GPU_LAUNCH_STREAM_ASYNC:
				cl = &(clients[req->instance_id]);
				tid = req->thread_id;

				cl->worker_thread_num = cl->gpu_info->thread_num;
				blk_num = cl->blk_num;
#if !defined(SYNC_MODE)
				record_blk_num[cl->instance_id] = blk_num;
				record_blk_num_thread[cl->instance_id][tid] = blk_num;
#endif
				threads_per_blk = cl->threads_per_blk;
				if (blk_num <= 0 || threads_per_blk < 128 || threads_per_blk > 1024) {
					rte_exit(EXIT_FAILURE, "instance id %d, blk_num %d, threads_per_blk %d\n", cl->instance_id, blk_num, threads_per_blk);
				}

				if (blk_num > SM_TOTAL_NUM - allocated_sm) {
					rte_exit(EXIT_FAILURE, "There should always have available SMs\n");
				}

				allocated_sm += blk_num;
				cl->sync[tid] = 0;

				char *args = cl->gpu_info->args[tid];
				uint64_t *arg_info = (uint64_t *)(cl->gpu_info->arg_info[tid]);
				uint64_t arg_num = arg_info[0];

				for (i = 0; i < arg_num; i ++) {
					offset = arg_info[1 + i];
					arg_info[1 + i] = (uint64_t)((uint8_t *)args + offset);
				}

				checkCudaErrors( cuEventRecord(cl->kern_start[tid], cl->stream[tid]) );
				checkCudaErrors( cuLaunchKernel(cl->function, 
							blk_num, 1, 1,  // Nx1x1 blocks
							threads_per_blk, 1, 1, // Mx1x1 threads
							0, cl->stream[tid], (void **)&(arg_info[1]), 0) );
				checkCudaErrors( cuEventRecord(cl->kern_end[tid], cl->stream[tid]) );

				rte_mempool_put(nf_request_pool, req);
				break;

			case REQ_GPU_LAUNCH_ALL_STREAM:
				cl = &(clients[req->instance_id]);
				/* The thread_num in gpu_info can be modified by clients.
				 * We record it in cl->worker_thread_num, so that the checking is correct in stream_callback */
				cl->worker_thread_num = cl->gpu_info->thread_num;
				/* Read the blk_num first, avoiding it being changed in the middle */
				blk_num = cl->blk_num;
			#if !defined(SYNC_MODE)
				record_blk_num[cl->instance_id] = blk_num;
			#endif
				threads_per_blk = cl->threads_per_blk;

				if (blk_num <= 0 || threads_per_blk < 128 || threads_per_blk > 1024) {
					rte_exit(EXIT_FAILURE, "instance id %d, blk_num %d, threads_per_blk %d\n", cl->instance_id, blk_num, threads_per_blk);
				}

				/* Remove the lock if stream_callback is not called */
				//pthread_mutex_lock(&lock);

			#if defined(MEASURE_KERNEL_TIME)
				checkCudaErrors( cuCtxSynchronize() );
				struct timespec start, end;
				clock_gettime(CLOCK_MONOTONIC, &start);
			#endif

#if defined(UNCO_SHARE_GPU) || defined(FAIR_SHARE_GPU)
				if (1) {
#else
				if (blk_num * cl->worker_thread_num <= SM_TOTAL_NUM - allocated_sm) {
#endif
					/* Enough SMs left */
					allocated_sm += blk_num * cl->worker_thread_num;
					//pthread_mutex_unlock(&lock);

					char *args;
					uint64_t *arg_info;
					uint64_t arg_num;

					RTE_LOG(DEBUG, APP, "cuLaunchKernel: (allocated_sm %d, total sm %d)\n", allocated_sm, SM_TOTAL_NUM);

					for (tid = 0; tid < cl->worker_thread_num; tid ++) {
						/* NOTE: this loop should be performed before the following loop, or the callback will execute first 
						 * before the next sync is set, leading to bugs */
						cl->sync[tid] = 0;
					}

					for (tid = 0; tid < cl->worker_thread_num; tid ++) {
						args = cl->gpu_info->args[tid];
						arg_info = (uint64_t *)(cl->gpu_info->arg_info[tid]);
						arg_num = arg_info[0];

						RTE_LOG(DEBUG, APP, "\t thread_id: %d, arg_num: %ld\n", tid, arg_num);

						/* Transfer all argument offsets into pointers */
						for (i = 0; i < arg_num; i ++) {
							offset = arg_info[1 + i];
							RTE_LOG(DEBUG, APP, "\t\t%ld: offset %ld, tid = %d\n", i, offset, tid);
							arg_info[1 + i] = (uint64_t)((uint8_t *)args + offset);
							RTE_LOG(DEBUG, APP, "\t\t arg%ld: %lx\n", i, *(uint64_t *)((uint8_t *)args + offset));
						}

						/* Launch Kernel */
						checkCudaErrors( cuLaunchKernel(cl->function, 
									blk_num, 1, 1,  // Nx1x1 blocks
									threads_per_blk, 1, 1, // 1x1x1 threads
									0, cl->stream[tid], (void **)&(arg_info[1]), 0) );
						
					#if defined(GRAPH_TIME) || defined(SYNC_MODE)
						clock_gettime(CLOCK_MONOTONIC, &tt);

						/* FIXME: only works for one thread per NF */
						cl->stats.kernel_start = (double)1000000*tt.tv_sec + tt.tv_nsec/1000;

						/* Add callback */
						type_arg a;
						a.info.instance_id = cl->instance_id;
						a.info.blk_num = blk_num;
						checkCudaErrors( cuStreamAddCallback(cl->stream[tid], stream_callback, (void *)a.arg, 0) );
					#endif
					#if defined(GRAPH_TIME)
						printf("%d\t3\t%.2lf\n", cl->instance_id, (double)1000000*tt.tv_sec + tt.tv_nsec/1000);
					#endif
					}

					/* Release the request */
					rte_mempool_put(nf_request_pool, req);

				} else {
					rte_exit(EXIT_FAILURE, "There should always have available SMs\n");
					pthread_mutex_unlock(&lock);
					/* Not enough SMs left, wait until enough SMs are released by other NFs.
					 * Assume each NF will wait for our response message. */
					if (pending_req_header == NULL) {
						pending_req_header = pending_req_tail = req;
					} else {
						assert(pending_req_tail != NULL);
						pending_req_tail->next = req;
						pending_req_tail = req;
					}
				}
			#if defined(NOT_SHARE_GPU) || defined(MEASURE_KERNEL_TIME)
				checkCudaErrors( cuCtxSynchronize() );
			#endif

			#if defined(MEASURE_KERNEL_TIME)
				clock_gettime(CLOCK_MONOTONIC, &end);
				double diff = 1000000 * (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/1000;
				//printf("[%d] Kernel time is %.2lf, worker_number is %d, blk_num %d, threads_per_blk %d\n",
				//		total_kernel_time_cnt, diff, cl->worker_thread_num, cl->blk_num, cl->threads_per_blk);
				total_kernel_time_diff += diff;
				total_kernel_time_cnt ++;
			#endif
				break;

			case REQ_GPU_SYNC:
				/* Synchronize all the streams for the NF */
				cl = &(clients[req->instance_id]);

				for (tid = 0; tid < cl->worker_thread_num; tid ++) {
					cl->sync[tid] = 0;
				}

				for (tid = 0; tid < cl->worker_thread_num; tid ++) {
					checkCudaErrors( cuStreamAddCallback(cl->stream[tid], sync_callback, (void *)(uint64_t)(cl->instance_id), 0) );
				}

				/* Release allocated SMs */
			#if !defined(SYNC_MODE)
				allocated_sm -= record_blk_num[cl->instance_id] * cl->worker_thread_num;
			#endif

				rte_mempool_put(nf_request_pool, req);
				break;

			case REQ_GPU_SYNC_STREAM:
				cl = &(clients[req->instance_id]);
				tid = req->thread_id;

				checkCudaErrors( cuEventRecord(cl->gpu_end[tid], cl->stream[tid]) );
				checkCudaErrors( cuStreamAddCallback(cl->stream[tid], stream_sync_callback, (void *)(req), 0) );
			#if !defined(SYNC_MODE)
				allocated_sm -= record_blk_num_thread[cl->instance_id][tid];
			#endif
				break;

			case REQ_GPU_RECORD_START:
				cl = &(clients[req->instance_id]);
				tid = req->thread_id;

				checkCudaErrors( cuEventRecord(cl->gpu_start[tid], cl->stream[tid]) );
				rte_mempool_put(nf_request_pool, req);
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
