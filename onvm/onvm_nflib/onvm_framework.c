#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>
#include <emmintrin.h>
#include <signal.h>

#include "onvm_framework.h"
#include "onvm_nflib.h"
#include "onvm_includes.h"
#include "fifo.h"

extern struct rte_mempool *nf_request_mp;
extern struct rte_mempool *nf_response_mp;
extern struct rte_ring *nf_request_queue;
extern struct onvm_nf_info *nf_info;
extern struct client_tx_stats *tx_stats;
extern volatile uint8_t keep_running;
extern void onvm_nflib_handle_signal(int sig);

/* shared data from server. */
struct gpu_schedule_info *gpu_info;

typedef struct pseudo_struct_s {
	int64_t job_num;
} pseudo_struct_t;

static nfv_batch_t batch_set[MAX_CPU_THREAD_NUM];
static void *(*INIT_FUNC)(void);
static void (*BATCH_FUNC)(void *,  struct rte_mbuf *);
static void (*POST_FUNC)(void *, struct rte_mbuf *, int);

#if defined(BQUEUE_SWITCH)
static int onvm_framework_cpu(int thread_id);
#else
static int onvm_framework_cpu(struct rte_mbuf **pkt, int rcv_pkt_num, int thread_id);
#endif

static pthread_key_t my_batch;
static pthread_mutex_t lock;

#define USE_LOCK 1

/* NOTE: The batch size should be at least 10x? larger than the number of items 
 * in PKTMBUF_POOL when running local. Or not enough mbufs to loop */
static int BATCH_SIZE = 1024;
static volatile int launch_kernel = 0;

int NF_REQUIRED_LATENCY = 1000; // us -- default latency
int INIT_WORKER_THREAD_NUM = 1; // us -- default latency

struct thread_arg {
	int thread_id;
};

static int
gpu_get_available_buf_id(nfv_batch_t *batch)
{
	int id;

	/* Because this is called always after mega_gpu_give_to_sender(), 
	 * There will always be at least one available buf for receiver */
	//assert(batch->available_buf_id[0] != -1);

#if defined(USE_LOCK)
	pthread_mutex_lock(&(batch->mutex_available_buf_id));
	id = batch->available_buf_id[0];
	batch->available_buf_id[0] = batch->available_buf_id[1];
	batch->available_buf_id[1] = -1; 
	pthread_mutex_unlock(&(batch->mutex_available_buf_id));
#else
	if (batch->available_buf_id[0] != -1) {
		id = batch->available_buf_id[0];
		batch->available_buf_id[0] = -1;
	} else if (batch->available_buf_id[1] != -1) {
		id = batch->available_buf_id[1];
		batch->available_buf_id[1] = -1;
	} else {
		assert(0);
	}
#endif
	return id;
}

static int
gpu_get_batch(nfv_batch_t *batch_set)
{
	unsigned int i;
	int available_buf_id;
	nfv_batch_t *batch;

	/* Tell the CPU worker we are taking the batch */
	for (i = 0; i < gpu_info->thread_num; i ++) {
		batch = &(batch_set[i]);

		assert(batch->gpu_buf_id == -1);

		available_buf_id = gpu_get_available_buf_id(batch);

		batch->gpu_buf_id = batch->receiver_buf_id;

		/* Let the receiver know the new available buffer transparently */
		batch->receiver_buf_id = available_buf_id;
	}
	return batch->gpu_buf_id;
}

/* Tell the CPU sender that this batch has been completed */
static void
gpu_give_to_sender(nfv_batch_t *batch_set)
{
	unsigned int i;
	nfv_batch_t *batch;

	for (i = 0; i < gpu_info->thread_num; i ++) {
		batch = &(batch_set[i]);

		if (batch->sender_buf_id != -1) {
			RTE_LOG(DEBUG, APP, "Post processing not completed while GPU completes processing\n");
		}
		/* Wait for the sender to complete last batch forwarding */
		while ((batch->sender_buf_id != -1) && keep_running) ;

		/* Give the buf to sender */
		batch->sender_buf_id = batch->gpu_buf_id;
		batch->gpu_buf_id = -1;
	}

	return ;
}

static int
sender_give_available_buffer(void)
{
	nfv_batch_t *batch = pthread_getspecific(my_batch);
	batch->post_idx = 0;

	pseudo_struct_t *buf = (pseudo_struct_t *)(batch->user_bufs[batch->sender_buf_id]);
	buf->job_num = 0;

	//printf("<<< [sender %d] < give available buffer %d\n", batch->thread_id, batch->sender_buf_id);
	/* tell the receiver that the buffer is available */
#if defined(USE_LOCK)
	pthread_mutex_lock(&(batch->mutex_available_buf_id));
	if (batch->available_buf_id[0] == -1) {
		batch->available_buf_id[0] = batch->sender_buf_id;
	} else if (batch->available_buf_id[1] == -1) {
		batch->available_buf_id[1] = batch->sender_buf_id;
	} else {
		rte_exit(EXIT_FAILURE, "Three buffers available \n");
	}
	pthread_mutex_unlock(&(batch->mutex_available_buf_id));
#else
	if (batch->available_buf_id[0] == -1) {
		batch->available_buf_id[0] = batch->sender_buf_id;
	} else if (batch->available_buf_id[1] == -1) {
		batch->available_buf_id[1] = batch->sender_buf_id;
	} else {
		rte_exit(EXIT_FAILURE, "Three buffers available \n");
	}
#endif

	batch->sender_buf_id = -1;

	return 0;
}

static void 
onvm_framework_thread_init(int thread_id)
{
	int i = 0;

	nfv_batch_t *batch = &(batch_set[thread_id]);
	/* The main thread set twice, elegant plan? */
	pthread_setspecific(my_batch, (void *)batch);
	batch->thread_id = thread_id;
	batch->post_idx = 0;

	batch->sender_buf_id = -1;
	batch->gpu_buf_id = -1;
	batch->receiver_buf_id = 0;
	batch->available_buf_id[0] = 1;
	batch->available_buf_id[1] = 2;

	assert(pthread_mutex_init(&(batch->mutex_sender_buf_id), NULL) == 0);
	assert(pthread_mutex_init(&(batch->mutex_available_buf_id), NULL) == 0);
	assert(pthread_mutex_init(&(batch->mutex_batch_launch), NULL) == 0);

	/* the last 1 is used to mark the allocation for not the first thread */
	if (thread_id != 0) {
		gcudaAllocSize(0, 0, 1);
	}

	for (i = 0; i < NUM_BATCH_BUF; i ++) {
		batch->user_bufs[i] = INIT_FUNC();
		batch->pkt_ptr[i] = (struct rte_mbuf **)malloc(sizeof(struct rte_mbuf *) * MAX_BATCH_SIZE);
	}
}

static int 
cpu_thread(void *arg)
{
	struct thread_arg *my_arg = (struct thread_arg *)arg;
	unsigned cur_lcore = rte_lcore_id();

	onvm_framework_thread_init(my_arg->thread_id);

	pthread_mutex_lock(&lock);
	gpu_info->thread_num ++;
	pthread_mutex_unlock(&lock);

	RTE_LOG(INFO, APP, "New CPU thread %d is spawned, running on lcore %u, total_thread %d\n", my_arg->thread_id, cur_lcore, gpu_info->thread_num);

	onvm_nflib_run(&(onvm_framework_cpu), my_arg->thread_id);

	RTE_LOG(INFO, APP, "Thread %d terminated on core %d\n", my_arg->thread_id, cur_lcore);

	return 0;
}

static void 
onvm_framework_spawn_thread(int thread_id, unsigned core_id)
{
	struct thread_arg *arg = (struct thread_arg *)malloc(sizeof(struct thread_arg));
	arg->thread_id = thread_id;

	if (rte_eal_remote_launch(cpu_thread, (void *)arg, core_id) == -EBUSY) {
		rte_exit(EXIT_FAILURE, "Core %d is busy, cannot allocate to run threads\n", core_id);
	}
}

#if defined(BQUEUE_SWITCH)
static int
onvm_framework_cpu(int thread_id)
{
	int i, buf_id;
	pseudo_struct_t *buf;
	nfv_batch_t *batch = (nfv_batch_t *)pthread_getspecific(my_batch);
	struct rte_mbuf *pkt;
	struct queue_t *tx_bqueue, *rx_bqueue;
	const struct rte_memzone *mz;
	int res;
	int instance_id = nf_info->instance_id;

#if defined(MEASURE_LATENCY)
	int latency_mark = 0;
	struct timespec start, end;
#endif

	mz = rte_memzone_lookup(get_rx_bq_name(instance_id, thread_id));
	if (mz == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get tx info structure\n");
	rx_bqueue = (struct queue_t *)(mz->addr);

	mz = rte_memzone_lookup(get_tx_bq_name(instance_id, thread_id));
	if (mz == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get tx info structure\n");
	tx_bqueue = (struct queue_t *)(mz->addr);

	for (; keep_running;) {

		res = bq_dequeue(rx_bqueue, &pkt);
		if (unlikely(res != SUCCESS))
			goto send;
		assert(pkt != NULL);

#if defined(NF_RX_SPEED_TEST)
		rte_pktmbuf_free(pkt);
		continue;
#endif
again:
		/* Read the buf_id for each pkt, in case that the GPU scheduler
		 * thread changes the id. */
		buf_id = batch->receiver_buf_id;
		buf = (pseudo_struct_t *)(batch->user_bufs[buf_id]);

#if defined(MEASURE_LATENCY)
		if (thread_id == 0 && buf->job_num == 0 && latency_mark == 0) {
			latency_mark = 1;
			clock_gettime(CLOCK_MONOTONIC, &start);
		}
#endif

		if ((buf->job_num >= B_PARA * BATCH_SIZE) && (thread_id == 0) && (launch_kernel == 0)) {
			launch_kernel = 1;
		}

#if defined(BATCH_DRIVEN_BUFFER_PASS)
		if (unlikely(buf->job_num >= BATCH_SIZE)) {
			/* This nf_drop statistic info is important for scheduling, cannot remove */
			if (thread_id == 0)
				tx_stats[instance_id].nf_drop ++;
			rte_pktmbuf_free(pkt);
			goto send;
		}
#else
		if (unlikely(buf->job_num >= MAX_BATCH_SIZE)) {
			/* Batch is full, drop the packets */
			if (thread_id == 0)
				tx_stats[instance_id].nf_drop ++;
			rte_pktmbuf_free(pkt);
			goto send;
		}
#endif

		//RTE_LOG(INFO, APP, "buf_id %d, thread_id %d, buf->job_num %d\n", buf_id, batch->thread_id, buf->job_num);
		BATCH_FUNC((void *)buf, pkt);
		batch->pkt_ptr[buf_id][buf->job_num] = pkt;
		/* the first element is required to be the job_num */
		buf->job_num ++;

		if (unlikely(buf_id != batch->receiver_buf_id)) {
			RTE_LOG(DEBUG, APP, "Buffer is switched during insertion\n");
			buf->job_num --;
			if (buf->job_num < 0) buf->job_num = 0;
			goto again;
		}

send:
		/* Post Processing */
		if (batch->sender_buf_id == -1) {
			/* The send buf has been processed */
			continue;
		}

		buf = (pseudo_struct_t *)(batch->user_bufs[batch->sender_buf_id]);

		/* Receive one packet, post process two packets */
		for (i = 0; (i < 2) && (batch->post_idx < buf->job_num); i ++, batch->post_idx ++) {
			POST_FUNC((void *)buf, batch->pkt_ptr[batch->sender_buf_id][batch->post_idx], batch->post_idx);

#if defined(MEASURE_LATENCY)
			if (thread_id == 0 && batch->post_idx == 0 && latency_mark == 1) {
				clock_gettime(CLOCK_MONOTONIC, &end);
				printf("%.2lf\n", (double)(1000000 * (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/1000));
				latency_mark = 0;
			}
#endif

			res = bq_enqueue(tx_bqueue, batch->pkt_ptr[batch->sender_buf_id][batch->post_idx]);
			if (unlikely(res != SUCCESS)) {
				if (thread_id == 0)
					tx_stats[instance_id].nf_drop_enq ++;
				rte_pktmbuf_free(batch->pkt_ptr[batch->sender_buf_id][batch->post_idx]);
			}
			//if (thread_id == 0)
			//	tx_stats[instance_id].tx ++;
		}

		if (unlikely(batch->post_idx == buf->job_num)) {
			sender_give_available_buffer();
		}
	}

	return 0;
}

#else /* BQUEUE_SWITCH */

static int
onvm_framework_cpu(struct rte_mbuf **pkts, int rcv_pkt_num, int thread_id)
{
	int i, buf_id;
	pseudo_struct_t *buf;
	nfv_batch_t *batch = (nfv_batch_t *)pthread_getspecific(my_batch);
	struct rte_mbuf *pkt;
	//int instance_id = nf_info->instance_id;

	/* Batch the received packets one by one */
	for (i = 0; i < rcv_pkt_num; i ++) {
		pkt = pkts[i];
again:
		/* Read the buf_id for each pkt, in case that the GPU scheduler
		 * thread changes the id. */
		buf_id = batch->receiver_buf_id;
		buf = (pseudo_struct_t *)(batch->user_bufs[buf_id]);

		if ((buf->job_num >= B_PARA * BATCH_SIZE) && (thread_id == 0) && (launch_kernel == 0)) {
			launch_kernel = 1;
		}

		if (buf->job_num >= MAX_BATCH_SIZE) {
			/* Batch is full, drop the packets */
			//if (thread_id == 0)
			//	tx_stats[instance_id].nf_drop ++;
			rte_pktmbuf_free(pkt);
			continue;
		}

		//RTE_LOG(INFO, APP, "buf_id %d, thread_id %d, buf->job_num %d\n", buf_id, batch->thread_id, buf->job_num);
		BATCH_FUNC((void *)buf, pkt);
		batch->pkt_ptr[buf_id][buf->job_num] = pkt;
		/* the first element is required to be the job_num */
		buf->job_num ++;

		if (buf_id != batch->receiver_buf_id) {
			RTE_LOG(DEBUG, APP, "Buffer is switched during insertion\n");
			buf->job_num --;
			if (buf->job_num < 0) buf->job_num = 0;
			goto again;
		}
	}

	if (batch->sender_buf_id == -1) {
		/* The send buf has been processed */
		return 0;
	}

	/* Adjusting the number of packets to be processed in post-processing */
	buf = (pseudo_struct_t *)(batch->user_bufs[batch->sender_buf_id]);
	/* post_pkt_batch should be set to a reasonable value
	 * 1) even receive no packets, it should process send_buf 
	 * 2) process the send_buf before accumulating BATCH_SIZE pkts in recv_buf */
	int post_pkt_batch = PKT_READ_SIZE * 2 - rcv_pkt_num;

	/* post processing */
	for (i = 0; (i < post_pkt_batch) && (batch->post_idx < buf->job_num); i ++, batch->post_idx ++) {
		POST_FUNC((void *)buf, batch->pkt_ptr[batch->sender_buf_id][batch->post_idx], batch->post_idx);
	}

	if (batch->post_idx == buf->job_num) {
		/* Send the batch out */
		if (buf->job_num != 0) {
			/* TODO: send out in small batches */
			onvm_nflib_send_processed(batch->pkt_ptr[batch->sender_buf_id], buf->job_num, thread_id);
		}

		sender_give_available_buffer();
	}

	return 0;
}
#endif /* BQUEUE_SWITCH */

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

	gpu_info->thread_num = 0;
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
onvm_framework_start_cpu(void *(*user_init_buf_func)(void), 
						void (*user_batch_func)(void *,  struct rte_mbuf *),
						void (*user_post_func)(void *, struct rte_mbuf *, int))
{
	INIT_FUNC = user_init_buf_func;
	BATCH_FUNC = user_batch_func;
	POST_FUNC = user_post_func;

	int i;
    unsigned cur_lcore = rte_lcore_id();
	for (i = 0; i < INIT_WORKER_THREAD_NUM; i ++) {
		/* Better to wait for a while between launching two threads, don't know why */
		sleep(1);
        cur_lcore =	rte_get_next_lcore(cur_lcore, 1, 1);
		onvm_framework_spawn_thread(i, cur_lcore);
	}
}

void
onvm_framework_start_gpu(void (*user_gpu_htod)(void *, unsigned int),
						void (*user_gpu_dtoh)(void *, unsigned int),
						void (*user_gpu_set_arg)(void *, void *, void *))
{
	int gpu_buf_id;
	int instance_id = nf_info->instance_id;
	pseudo_struct_t *buf;
	unsigned int i;
	struct timespec start, end;
	double diff;

	/* Listen for ^C and docker stop so we can exit gracefully */
	signal(SIGINT, onvm_nflib_handle_signal);
	signal(SIGTERM, onvm_nflib_handle_signal);

	if (user_gpu_set_arg == NULL || user_gpu_htod == NULL || user_gpu_dtoh == NULL) {
		rte_exit(EXIT_FAILURE, "GPU function is NULL\n");
	}

	while (gpu_info->thread_num != (unsigned int)INIT_WORKER_THREAD_NUM && keep_running) ;

	unsigned cur_lcore = rte_lcore_id();
	RTE_LOG(INFO, APP, "GPU thread is running on lcore %u\n", cur_lcore);
	printf("[Press Ctrl-C to quit ...]\n\n");

	for (; keep_running;) {
		/* 1. Wait until the batch size is reached for the first thread.
		 * We have load balance among all threads, so their batch size are the same. */
		RTE_LOG(DEBUG, APP, "GPU thread is launching kernel\n");

		while ((launch_kernel == 0) && keep_running) ;
		/* 2. Get Buffers from receivers */
		gpu_buf_id = gpu_get_batch(batch_set);
		launch_kernel = 0;

		for (i = 0; i < gpu_info->thread_num; i ++) {
			buf = (pseudo_struct_t *)(batch_set[i].user_bufs[gpu_buf_id]);
			tx_stats[instance_id].batch_size += buf->job_num;
			tx_stats[instance_id].batch_cnt ++;
		}

		clock_gettime(CLOCK_MONOTONIC, &start);
	#if defined(GRAPH_TIME)
		printf("%d\t7\t%.2lf\n", instance_id, (double)1000000*start.tv_sec + start.tv_nsec/1000);
	#endif

#if !defined(NO_GPU) && !defined(PKTGEN_FRAMEWORK)

		/* 3. Launch kernel - USER DEFINED */
		unsigned int i;
	#if defined(STREAM_LAUNCH)
		for (i = 0; i < gpu_info->thread_num; i ++) {
			user_gpu_htod(batch_set[i].user_bufs[gpu_buf_id], i);

			user_gpu_set_arg(batch_set[i].user_bufs[gpu_buf_id], gpu_info->args[i], gpu_info->arg_info[i]);
			gcudaLaunchKernel(i);

			user_gpu_dtoh(batch_set[i].user_bufs[gpu_buf_id], i);
		}
	#else
		for (i = 0; i < gpu_info->thread_num; i ++) {
			user_gpu_htod(batch_set[i].user_bufs[gpu_buf_id], i);
			user_gpu_set_arg(batch_set[i].user_bufs[gpu_buf_id], gpu_info->args[i], gpu_info->arg_info[i]);
		}

		gcudaLaunchKernel_allStream();

		for (i = 0; i < gpu_info->thread_num; i ++) {
			user_gpu_dtoh(batch_set[i].user_bufs[gpu_buf_id], i);
		}
	#endif

		/* 4. Explicit SYNC if commands are not executed in SYNC_MODE, wait for the kernels to complete */
	#if !defined(GRAPH_TIME) && !defined(SYNC_MODE)
		gcudaDeviceSynchronize();
	#endif
#endif

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = 1000000 * (end.tv_sec-start.tv_sec)+ (end.tv_nsec-start.tv_nsec)/1000;
	#if defined(GRAPH_TIME)
		printf("%d\t8\t%.2lf\n", instance_id, (double)1000000*end.tv_sec + end.tv_nsec/1000);
	#endif
		//buf = (pseudo_struct_t *)(batch_set[0].user_bufs[gpu_buf_id]);
		//printf("[%d] GPU time in framework: %.2lf, batch size %ld\n", gpu_buf_id, diff, buf->job_num);
		tx_stats[instance_id].gpu_time += diff;
		tx_stats[instance_id].gpu_time_cnt ++;

		/* 5. Pass the results to CPU again for post processing */
		gpu_give_to_sender(batch_set);

		RTE_LOG(DEBUG, APP, "Handle GPU processed results to sender\n");

		// TODO: check the tx status & send requests to the Manager
	}
}

/* ======================================= */

void
gcudaAllocSize(int size_per_thread, int size_global, int first)
{
	int size;
	nfv_batch_t *batch;
	static int size_thread;

	if (first == 0) {
		/* Main thread of this NF */
		size = size_per_thread * 3 + size_global;
		size_thread = size_per_thread;
		batch = &(batch_set[0]);
		pthread_key_create(&my_batch, NULL);
		pthread_setspecific(my_batch, (void *)batch);
	} else {
		/* Spawned thread of this NF */
		size = size_thread * 3;
		batch = (nfv_batch_t *)pthread_getspecific(my_batch);
	}

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_HOST_MALLOC;
	req->instance_id = nf_info->instance_id;
	req->thread_id = batch->thread_id;
	req->size = size;  /* 3 buffers in total */

	RTE_LOG(DEBUG, APP, "[%d] Host Alloc, size %d\n", batch->thread_id, size);

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	struct rte_ring *nf_response_ring = rte_ring_lookup(get_rsp_queue_name(nf_info->instance_id, batch->thread_id));
	if (nf_response_ring == NULL)
		rte_exit(EXIT_FAILURE, "Failed to get response ring\n");

	struct nf_rsp *rsp;
	while (rte_ring_dequeue(nf_response_ring, (void **)&rsp) != 0 && keep_running) ;
	assert((rsp->type == RSP_HOST_MALLOC) & (rsp->states == RSP_SUCCESS));

	const struct rte_memzone *mz = rte_memzone_lookup(get_buf_name(nf_info->instance_id, batch->thread_id));
	if (mz == NULL)
		rte_exit(EXIT_FAILURE, "Cannot find memzone\n");

	batch->host_mem_addr_base = mz->addr;
	batch->host_mem_addr_cur = mz->addr;
	batch->host_mem_size_total = mz->len;
	batch->host_mem_size_left = mz->len;

	rte_mempool_put(nf_response_mp, rsp);
}

void
gcudaMalloc(CUdeviceptr *p, int size)
{
	nfv_batch_t *batch = (nfv_batch_t *)pthread_getspecific(my_batch);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_GPU_MALLOC;
	req->instance_id = nf_info->instance_id;
	req->thread_id = batch->thread_id;
	req->size = size;

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	struct rte_ring *nf_response_ring = rte_ring_lookup(get_rsp_queue_name(nf_info->instance_id, batch->thread_id));
	if (nf_response_ring == NULL)
		rte_exit(EXIT_FAILURE, "Failed to get response ring\n");

	struct nf_rsp *rsp;
	while (rte_ring_dequeue(nf_response_ring, (void **)&rsp) != 0 && keep_running) ;

	assert((rsp->type == RSP_GPU_MALLOC) & (rsp->states == RSP_SUCCESS));
	*p = rsp->dev_ptr;

	RTE_LOG(DEBUG, APP, "[%d] cudaMalloc %lx, size %d\n", batch->thread_id, (uint64_t)*p, size);

	rte_mempool_put(nf_response_mp, rsp);
}

void
gcudaHostAlloc(void **p, int size)
{
	nfv_batch_t *batch = (nfv_batch_t *)pthread_getspecific(my_batch);

	if (size > batch->host_mem_size_left)
		rte_exit(EXIT_FAILURE, "[%d] No enough host memory space left %d > %d\n", batch->thread_id, size, batch->host_mem_size_left);

	*p = batch->host_mem_addr_cur;
	batch->host_mem_addr_cur = (void *)((char *)batch->host_mem_addr_cur + size);
	batch->host_mem_size_left -= size;

	RTE_LOG(DEBUG, APP, "[%d] allocating %d host memory, leaving %d\n", batch->thread_id, size, batch->host_mem_size_left);
}

void
gcudaMemcpyHtoD(CUdeviceptr dst, void *src, int size, int sync, unsigned int thread_id)
{
	nfv_batch_t *batch = &(batch_set[thread_id]);
	assert(batch->thread_id == thread_id);

#if defined(GRAPH_TIME) || defined(SYNC_MODE)
	sync = SYNC;
#endif

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	if (sync == SYNC) {
		req->type = REQ_GPU_MEMCPY_HTOD_SYNC;
	} else {
		req->type = REQ_GPU_MEMCPY_HTOD_ASYNC;
	}
	req->device_ptr = dst;
	req->host_offset = (char *)src - (char *)(batch->host_mem_addr_base);
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;
	req->size = size;

	RTE_LOG(DEBUG, APP, "[%d] cudaMemcpyHtoD, dst %lx, host offset %d, size %d, sync %d\n", 
			thread_id, (uint64_t)dst, req->host_offset, size, sync);

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	if (sync == SYNC) {
		struct rte_ring *nf_response_ring = rte_ring_lookup(get_rsp_queue_name(nf_info->instance_id, thread_id));
		if (nf_response_ring == NULL)
			rte_exit(EXIT_FAILURE, "Failed to get response ring\n");

		struct nf_rsp *rsp;
		while (rte_ring_dequeue(nf_response_ring, (void **)&rsp) != 0 && keep_running) ;
		assert((rsp->type == RSP_GPU_MEMCPY_HTOD_SYNC) & (rsp->states == RSP_SUCCESS));

		rte_mempool_put(nf_response_mp, rsp);
	}
}

void
gcudaMemcpyDtoH(void *dst, CUdeviceptr src, int size, int sync, unsigned int thread_id)
{
	nfv_batch_t *batch = &(batch_set[thread_id]);
	assert(batch->thread_id == thread_id);

#if defined(GRAPH_TIME) || defined(SYNC_MODE)
	sync = SYNC;
#endif

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	if (sync == SYNC) {
		req->type = REQ_GPU_MEMCPY_DTOH_SYNC;
	} else if (sync == ASYNC) {
		req->type = REQ_GPU_MEMCPY_DTOH_ASYNC;
	}
	req->device_ptr = src;
	req->host_offset = (char *)dst - (char *)(batch->host_mem_addr_base);
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;
	req->size = size;

	RTE_LOG(DEBUG, APP, "[%d] cudaMemcpyDtoH, host offset %d, src %lx, size %d\n", 
			thread_id, req->host_offset, (uint64_t)src, size);

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	if (sync == SYNC) {
		struct rte_ring *nf_response_ring = rte_ring_lookup(get_rsp_queue_name(nf_info->instance_id, thread_id));
		if (nf_response_ring == NULL)
			rte_exit(EXIT_FAILURE, "Failed to get response ring\n");

		struct nf_rsp *rsp;
		while (rte_ring_dequeue(nf_response_ring, (void **)&rsp) != 0 && keep_running) ;
		assert((rsp->type == RSP_GPU_MEMCPY_DTOH_SYNC) & (rsp->states == RSP_SUCCESS));

		rte_mempool_put(nf_response_mp, rsp);
	}
}

void
gcudaLaunchKernel(int thread_id)
{
	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_LAUNCH_STREAM_ASYNC;
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;

	RTE_LOG(DEBUG, APP, "[G] cudaLaunchKernel\n");

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

static void
gcudaWaitForSyncResponse(void)
{
	/* Waiting for the manager to send back the SYNC response */
	struct rte_ring *nf_response_ring = rte_ring_lookup(get_rsp_queue_name(nf_info->instance_id, GLOBAL_RSP_QUEUE));
	if (nf_response_ring == NULL)
		rte_exit(EXIT_FAILURE, "Failed to get response ring\n");

	struct nf_rsp *rsp;
	while (rte_ring_dequeue(nf_response_ring, (void **)&rsp) != 0 && keep_running) ;

	if ((rsp->type != RSP_GPU_GLOBAL_SYNC) && (rsp->type != RSP_GPU_KERNEL_SYNC))
		rte_exit(EXIT_FAILURE, "Wrong response type %d\n", rsp->type);

	if ((rsp->batch_size != 0) && (rsp->batch_size < MAX_BATCH_SIZE)) {
		if (BATCH_SIZE != rsp->batch_size) {
			BATCH_SIZE = rsp->batch_size;
			RTE_LOG(DEBUG, APP, "Update BATCH_SIZE as %d\n", BATCH_SIZE);
		}
	} else {
		RTE_LOG(DEBUG, APP, "Batch size is %d, larger than %d\n", rsp->batch_size, MAX_BATCH_SIZE);
		BATCH_SIZE = MAX_BATCH_SIZE;
	}

	rte_mempool_put(nf_response_mp, rsp);
}

void
gcudaLaunchKernel_allStream(void)
{
	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_LAUNCH_ALL_STREAM;
	req->instance_id = nf_info->instance_id;

	RTE_LOG(DEBUG, APP, "[G] cudaLaunchKernel_allStream\n");

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

#if defined(GRAPH_TIME) || defined(SYNC_MODE)
	gcudaWaitForSyncResponse();
#endif
}

void
gcudaDeviceSynchronize(void)
{
	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_GPU_SYNC;
	req->instance_id = nf_info->instance_id;

	RTE_LOG(DEBUG, APP, "[G] cudaDeviceSync\n");

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}

	gcudaWaitForSyncResponse();

	RTE_LOG(DEBUG, APP, "[G] cudaDeviceSync finished\n");
}
