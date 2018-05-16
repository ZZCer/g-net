
// CLEAR: 1

#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>
#include <emmintrin.h>
#include <signal.h>
#include <rte_malloc.h>

#include "onvm_framework.h"
#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_includes.h"

// ========== globals ==========
extern struct rte_mempool *nf_request_mp;
extern struct rte_mempool *nf_response_mp;
extern struct rte_ring *nf_request_queue;
extern struct onvm_nf_info *nf_info;
extern struct client *cl;
extern volatile uint8_t keep_running;
extern void onvm_nflib_handle_signal(int sig);

/* shared data from server. */
struct gpu_schedule_info *gpu_info;

int THREAD_NUM = 1;
int STREAM_NUM = 1;

pthread_key_t thread_local_key;

// ========== locals ==========
/* host memory area */
static void *host_mem_addr_base;
static void *host_mem_addr_cur;
static int host_mem_size_total;
static int host_mem_size_left;

/* gpu thread started */
static int all_threads_ready;

static nfv_batch_t batch_set[MAX_CONCURRENCY_NUM];

static pre_func_t  PRE_FUNC;
static post_func_t POST_FUNC;

static struct {
	int working;
} stream_ctx[MAX_CONCURRENCY_NUM];

static int onvm_framework_cpu(int thread_id);
static void gcudaSyncStreamRequest(void);
static int handleSyncStreamResponse(void);

static pthread_mutex_t lock;

/* NOTE: The batch size should be at least 10x? larger than the number of items 
 * in PKTMBUF_POOL when running local. Or not enough mbufs to loop */
static unsigned int BATCH_SIZE = 1024;

int NF_REQUIRED_LATENCY = 1000; // us -- default latency

struct thread_arg {
	int thread_id;
};

typedef struct thread_local_s {
	int thread_id;
	int stream_id;
} thread_local_t;

static inline int get_batch(int state) {
	int i;
	for (i = 0; i < THREAD_NUM + STREAM_NUM; i++) {
			if (batch_set[i].buf_state == state) return i;
	}
	return -1;
}

static inline int cpu_get_batch(void) {
       return get_batch(BUF_STATE_CPU_READY);
}

static inline int gpu_get_batch(void) {
       return get_batch(BUF_STATE_GPU_READY);
}

static int 
cpu_thread(void *arg)
{
	struct thread_arg *my_arg = (struct thread_arg *)arg;
	unsigned cur_lcore = rte_lcore_id();

	RTE_LOG(INFO, APP, "New CPU thread %d is spawned, running on lcore %u", my_arg->thread_id, cur_lcore);

	onvm_nflib_run(&(onvm_framework_cpu), my_arg->thread_id);

	RTE_LOG(INFO, APP, "Thread %d terminated on core %d\n", my_arg->thread_id, cur_lcore);

	return 0;
}

static void 
onvm_framework_spawn_thread(int thread_id)
{
	struct thread_arg *arg = (struct thread_arg *)malloc(sizeof(struct thread_arg));
	arg->thread_id = thread_id;

	unsigned cur_lcore = rte_lcore_id() + thread_id;
	cur_lcore =	rte_get_next_lcore(cur_lcore, 1, 1);
	if (rte_eal_remote_launch(cpu_thread, (void *)arg, cur_lcore) == -EBUSY) {
		rte_exit(EXIT_FAILURE, "Core %d is busy, cannot allocate to run threads\n", cur_lcore);
	}
}

static int
onvm_framework_cpu(int thread_id)
{
	int i, j;
	int buf_id;
	nfv_batch_t *batch;
	int cur_buf_size;
	struct rte_ring *rx_q, *tx_q;
	uint64_t starve_rx_counter = 0;
	uint64_t starve_gpu_counter = 0;

	rx_q = cl->rx_q_new;

	thread_local_t *local = (thread_local_t *)rte_calloc("thread_local", 1, sizeof(thread_local_t), 0);
	local->thread_id = thread_id;
	local->stream_id = -1;
	pthread_setspecific(thread_local_key, local);

	while (keep_running && !all_threads_ready);

	while (keep_running) {
		buf_id = cpu_get_batch();
		if (buf_id == -1) {
			starve_gpu_counter++;
			if (starve_gpu_counter == STARVE_THRESHOLD) {
				RTE_LOG(INFO, APP, "GPU starving\n");
			}
			continue;
		}
		if (starve_gpu_counter >= STARVE_THRESHOLD) {
			RTE_LOG(INFO, APP, "GPU resumed\n");
			starve_gpu_counter = 0;
		}
		batch = &batch_set[buf_id];
		cur_buf_size = batch->buf_size;

		// post-processing
		for (i = 0; i < cur_buf_size; i++) {
			POST_FUNC(batch->user_buf, batch->pkt_ptr[i], i);
		}

		// handle dropped packets
		for (i = j = 0; i < cur_buf_size; i++) {
			struct onvm_pkt_meta *meta = onvm_get_pkt_meta(batch->pkt_ptr[i]);
			if (meta->action != ONVM_NF_ACTION_DROP) {
				// swap
				struct rte_mbuf *p = batch->pkt_ptr[i];
				batch->pkt_ptr[i] = batch->pkt_ptr[j];
				batch->pkt_ptr[j++] = p;
			}
		}
		int num_packets = j;

		// tx
		tx_q = *(struct rte_ring * const volatile*)&cl->tx_q_new;
		int sent_packets = 0;
		if (likely(tx_q != NULL && num_packets != 0)) {
			sent_packets = rte_ring_enqueue_burst(tx_q, (void **)batch->pkt_ptr, num_packets, NULL);
		}
		if (sent_packets < cur_buf_size) {
			onvm_pkt_drop_batch(batch->pkt_ptr + sent_packets, cur_buf_size - sent_packets);
		}

		rte_spinlock_lock(&cl->stats.update_lock);
		cl->stats.tx += sent_packets;
		cl->stats.tx_drop += num_packets - sent_packets;
		cl->stats.act_drop += cur_buf_size - num_packets;
		rte_spinlock_unlock(&cl->stats.update_lock);

		// rx
		do {
			if (BATCH_SIZE != cl->batch_size) {
				BATCH_SIZE = cl->batch_size;
				RTE_LOG(INFO, APP, "batch size changed to %u\n", BATCH_SIZE);
			}
			num_packets = rte_ring_dequeue_bulk(rx_q, (void **)batch->pkt_ptr, BATCH_SIZE, NULL);
			if (num_packets == 0) {
				starve_rx_counter++;
				if (starve_rx_counter == STARVE_THRESHOLD) {
					RTE_LOG(INFO, APP, "Rx starving\n");
				}
			}
		} while (num_packets == 0);
		if (starve_rx_counter >= STARVE_THRESHOLD) {
			RTE_LOG(INFO, APP, "Rx resumed\n");
			starve_rx_counter = 0;
		}
		cur_buf_size = num_packets;
		batch->buf_size = cur_buf_size;

		// pre-processing
		uint64_t rx_datalen = 0;
		for (i = 0; i < cur_buf_size; i++) {
			rx_datalen += batch->pkt_ptr[i]->data_len;
			PRE_FUNC(batch->user_buf, batch->pkt_ptr[i], i);
		}

		rte_spinlock_lock(&cl->stats.update_lock);
		cl->stats.rx += cur_buf_size;		
		cl->stats.rx_datalen += rx_datalen;
		rte_spinlock_unlock(&cl->stats.update_lock);

		// launch kernel
		if (cur_buf_size > 0) {
			batch->buf_state = BUF_STATE_GPU_READY;
		}
	}

	return 0;
}

void
onvm_framework_install_kernel_perf_parameter(double k1, double b1, double k2, double b2)
{
	/* Parameters for linear equation between kernel execution time and batch size */
	gpu_info->k1[0] = k1;
	gpu_info->b1[0] = b1;
	/* Parameters for linear equation between data transfer time and batch size */
	gpu_info->k2[0] = k2;
	gpu_info->b2[0] = b2;

	gpu_info->para_num = 1;
}

void
onvm_framework_install_kernel_perf_para_set(double *u_k1, double *u_b1, double *u_k2, double *u_b2,
		unsigned int *pkt_size, unsigned int *line_start_batch, unsigned int para_num)
{
	unsigned int i;

	if (para_num > MAX_PARA_NUM)
		rte_exit(EXIT_FAILURE, "Too many kernel parameters installed\n");

	gpu_info->para_num = para_num;

	for (i = 0; i < para_num; i ++) {
		/* Parameters for linear equation between kernel execution time and batch size */
		gpu_info->k1[i] = u_k1[i];
		gpu_info->b1[i] = u_b1[i];
		/* Parameters for linear equation between data transfer time and batch size */
		gpu_info->k2[i] = u_k2[i];
		gpu_info->b2[i] = u_b2[i];

		gpu_info->pkt_size[i] = pkt_size[i];
		gpu_info->line_start_batch[i] = line_start_batch[i];
	}
}

void
onvm_framework_init(const char *module_file, const char *kernel_name)
{
	const struct rte_memzone *mz_gpu;

	/* This memzone has been initialized by the manager at start */
	mz_gpu = rte_memzone_lookup(get_gpu_info_name(nf_info->instance_id));
	if (mz_gpu == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get GPU info structre\n");
	gpu_info = mz_gpu->addr;

	gpu_info->stream_num = STREAM_NUM;
	gpu_info->init = 1;
	gpu_info->latency_us = NF_REQUIRED_LATENCY;
	gpu_info->launch_tx_thread = 0;
	gpu_info->launch_worker_thread = 0;

	if (pthread_mutex_init(&lock, NULL) != 0)
		rte_exit(EXIT_FAILURE, "Cannot init lock\n");

	/* For NF_FRAMEWORK, test only */
	if ((module_file == NULL) || (kernel_name == NULL)) {
		RTE_LOG(WARNING, APP, "module file or kernel name is NULL\n");
		return;
	}

	if ((strlen(module_file) >= MAX_MODULE_FILE_LEN) || (strlen(kernel_name) >= MAX_KERNEL_NAME_LEN))
		rte_exit(EXIT_FAILURE, "Name size is too large, %zu, %zu\n", strlen(module_file), strlen(kernel_name));

	rte_memcpy((void *)(gpu_info->module_file), module_file, strlen(module_file));
	rte_memcpy((void *)(gpu_info->kernel_name), kernel_name, strlen(kernel_name));
}

void
onvm_framework_start_cpu(init_func_t user_init_buf_func, pre_func_t user_pre_func, post_func_t user_post_func)
{
	PRE_FUNC = user_pre_func;
	POST_FUNC = user_post_func;

	all_threads_ready = 0;

	int i;
	for (i = 0; i < THREAD_NUM + STREAM_NUM; i++) {
		batch_set[i].buf_size = 0;
		batch_set[i].buf_state = BUF_STATE_CPU_READY;
		batch_set[i].user_buf = user_init_buf_func();
	}
	for (i = 0; i < THREAD_NUM; i ++) {
		onvm_framework_spawn_thread(i);
	}
}

void
onvm_framework_start_gpu(gpu_htod_t user_gpu_htod, gpu_dtoh_t user_gpu_dtoh, gpu_set_arg_t user_gpu_set_arg)
{
	int gpu_buf_id;
	int i;
	nfv_batch_t *batch;
	int stream_id;

	/* Listen for ^C and docker stop so we can exit gracefully */
	signal(SIGINT, onvm_nflib_handle_signal);

	if (user_gpu_set_arg == NULL || user_gpu_htod == NULL || user_gpu_dtoh == NULL) {
		rte_exit(EXIT_FAILURE, "GPU function is NULL\n");
	}

	thread_local_t *local = (thread_local_t *)rte_calloc("thread_local", 1, sizeof(thread_local_t), 0);
	local->thread_id = THREAD_NUM;
	pthread_setspecific(thread_local_key, local);

	for (i = 0; i < STREAM_NUM; i++) {
		local->stream_id = i;
		stream_ctx[i].working = -1;
		gcudaSyncStreamRequest();
	}

	unsigned cur_lcore = rte_lcore_id();
	RTE_LOG(INFO, APP, "GPU thread is running on lcore %u\n", cur_lcore);
	printf("[Press Ctrl-C to quit ...]\n\n");

	all_threads_ready = 1;

	for (; keep_running;) {
		RTE_LOG(DEBUG, APP, "GPU thread is launching kernel\n");

		// find an available stream
		do {
			stream_id = handleSyncStreamResponse();
		} while (stream_id == -1 && keep_running);
		if (!keep_running) break;
		local->stream_id = stream_id;

		// set previous work to finished
		if (stream_ctx[stream_id].working != -1) {
			batch_set[stream_ctx[stream_id].working].buf_state = BUF_STATE_CPU_READY;

			/*
			rte_spinlock_lock(&cl->stats.update_lock);
			cl->stats.batch_size += batch->buf_size;
			cl->stats.gpu_time += diff;
			cl->stats.batch_cnt++;
			rte_spinlock_unlock(&cl->stats.update_lock);
			*/
		}

		// find a batch
		do {
			gpu_buf_id = gpu_get_batch();
		} while (gpu_buf_id == -1 && keep_running);
		if (!keep_running) break;

		batch = &batch_set[gpu_buf_id];

		// go
		user_gpu_htod(batch->user_buf, batch->buf_size);
		user_gpu_set_arg(batch->user_buf, gpu_info->args[stream_id], gpu_info->arg_info[stream_id], batch->buf_size);
		gcudaLaunchKernel();
		user_gpu_dtoh(batch->user_buf, batch->buf_size);
		gcudaSyncStreamRequest();
	}

	onvm_nflib_stop(); // clean up
}

/* ======================================= */

void
gcudaAllocSize(int size_per_thread, int size_global)
{
	int size;
	size = size_per_thread * THREAD_NUM + size_global;

	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int thread_id = (local ? local->thread_id : 0);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_HOST_MALLOC;
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;
	req->size = size;

	RTE_LOG(DEBUG, APP, "[%d] Host Alloc, size %d\n", thread_id, size);

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	struct nf_rsp *rsp;
	while (rte_ring_dequeue(cl->response_q[thread_id], (void **)&rsp) != 0 && keep_running) ;
	assert((rsp->type == RSP_HOST_MALLOC) & (rsp->states == RSP_SUCCESS));

	const struct rte_memzone *mz = rte_memzone_lookup(get_buf_name(nf_info->instance_id));
	if (mz == NULL)
		rte_exit(EXIT_FAILURE, "Cannot find memzone\n");

	host_mem_addr_base = mz->addr;
	host_mem_addr_cur = mz->addr;
	host_mem_size_total = mz->len;
	host_mem_size_left = mz->len;

	rte_mempool_put(nf_response_mp, rsp);
}

void
gcudaMalloc(CUdeviceptr *p, int size)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int thread_id = (local ? local->thread_id : 0);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_GPU_MALLOC;
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;
	req->size = size;

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	struct nf_rsp *rsp;
	while (rte_ring_dequeue(cl->response_q[thread_id], (void **)&rsp) != 0 && keep_running) ;

	assert((rsp->type == RSP_GPU_MALLOC) & (rsp->states == RSP_SUCCESS));
	*p = rsp->dev_ptr;

	RTE_LOG(DEBUG, APP, "[%d] cudaMalloc %lx, size %d\n", thread_id, (uint64_t)*p, size);

	rte_mempool_put(nf_response_mp, rsp);
}

void
gcudaHostAlloc(void **p, int size)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int thread_id = (local ? local->thread_id : 0);

	if (size > host_mem_size_left)
		rte_exit(EXIT_FAILURE, "[%d] No enough host memory space left %d > %d\n", thread_id, size, host_mem_size_left);

	*p = host_mem_addr_cur;
	host_mem_addr_cur = (void *)((char *)host_mem_addr_cur + size);
	host_mem_size_left -= size;

	RTE_LOG(DEBUG, APP, "[%d] allocating %d host memory, leaving %d\n", thread_id, size, host_mem_size_left);
}

void
gcudaMemcpyHtoD(CUdeviceptr dst, void *src, int size)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int stream_id = (local ? local->stream_id : 0);
	assert(stream_id != -1);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_MEMCPY_HTOD_ASYNC;
	req->device_ptr = dst;
	req->host_offset = (char *)src - (char *)(host_mem_addr_base);
	req->instance_id = nf_info->instance_id;
	req->stream_id = stream_id;
	req->size = size;

	RTE_LOG(DEBUG, APP, "[s%d] cudaMemcpyHtoD, dst %lx, host offset %d, size %d", 
			stream_id, (uint64_t)dst, req->host_offset, size);

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

void
gcudaMemcpyDtoH(void *dst, CUdeviceptr src, int size)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int stream_id = (local ? local->stream_id : 0);
	assert(stream_id != -1);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_MEMCPY_DTOH_ASYNC;
	req->device_ptr = src;
	req->host_offset = (char *)dst - (char *)(host_mem_addr_base);
	req->instance_id = nf_info->instance_id;
	req->stream_id = stream_id;
	req->size = size;

	RTE_LOG(DEBUG, APP, "[s%d] cudaMemcpyDtoH, host offset %d, src %lx, size %d\n", 
			stream_id, req->host_offset, (uint64_t)src, size);

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

void
gcudaLaunchKernel(void)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int stream_id = (local ? local->stream_id : 0);
	assert(stream_id != -1);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_LAUNCH_STREAM_ASYNC;
	req->instance_id = nf_info->instance_id;
	req->stream_id = stream_id;

	RTE_LOG(DEBUG, APP, "[G] cudaLaunchKernel\n");

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

static void
gcudaSyncStreamRequest(void)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int thread_id = (local ? local->stream_id : 0);
	int stream_id = (local ? local->stream_id : 0);
	assert(stream_id != -1);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_SYNC_STREAM;
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;
	req->stream_id = stream_id;

	RTE_LOG(DEBUG, APP, "[G] cudaSyncStream\n");

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

static int
handleSyncStreamResponse(void)
{
	thread_local_t *local = (thread_local_t *)pthread_getspecific(thread_local_key);
	int thread_id = (local ? local->stream_id : 0);

	struct nf_rsp *rsp;
	if (0 == rte_ring_dequeue(cl->response_q[thread_id], (void **)&rsp))
		return -1;
	assert((rsp->type == RSP_GPU_SYNC_STREAM) & (rsp->states == RSP_SUCCESS));
	int stream_id = rsp->stream_id;
	rte_mempool_put(nf_response_mp, rsp);
	return stream_id;	
}