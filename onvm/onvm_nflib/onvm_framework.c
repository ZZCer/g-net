
// CLEAR: 1

#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>
#include <emmintrin.h>
#include <signal.h>

#include "onvm_framework.h"
#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_includes.h"

#include "../onvm_mgr/onvm_init.h"
#include "../onvm_mgr/pstack.h"

// #define ENABLE_PSTACK
// 这些extern变量都定义在onvm_init onvm_common onvm_nflib
extern struct rte_mempool *nf_request_mp;
extern struct rte_mempool *nf_response_mp;
extern struct rte_ring *nf_request_queue;
extern struct onvm_nf_info *nf_info;
extern struct client *cl;
extern uint16_t sync_plan;
extern volatile uint8_t keep_running;
extern void onvm_nflib_handle_signal(int sig);

pstack_thread_info pstack_info;


/* shared data from server. */
struct gpu_schedule_info *gpu_info;

typedef struct pseudo_struct_s {
	int64_t job_num;
} pseudo_struct_t;

//如果只在这个文件中使用的话，那么是如何获取对应数据的
static nfv_batch_t batch_set[MAX_CPU_THREAD_NUM];

static init_func_t INIT_FUNC;
static pre_func_t  PRE_FUNC;
static post_func_t POST_FUNC;
static cpu_batch_handle CPU_BATCH_FUNC;

//当程序仅仅只跑在cpu上面时，使用cpu_only模块
static int onvm_framework_cpu_only(int thread);
static int onvm_framework_cpu(int thread_id);
static void gcudaRecordStart(int thread_id);
static void gcudaStreamSynchronize(int thread_id);
static int gcudaPollForStreamSyncResponse(int thread_id);

static pthread_key_t my_batch;
static pthread_mutex_t lock;

static volatile int recv_token, send_token;

//用来表示nf的处理类型
extern int nf_handle_tag;

/* NOTE: The batch size should be at least 10x? larger than the number of items 
 * in PKTMBUF_POOL when running local. Or not enough mbufs to loop */
static int BATCH_SIZE = 1024;

int NF_REQUIRED_LATENCY = 1000; // us -- default latency
int INIT_WORKER_THREAD_NUM = 1;

unsigned last_assigned_core_id = 0;
static __thread int current_rx_qid;

struct thread_arg {
	int thread_id;
};


static inline int gpu_get_batch(nfv_batch_t *batch) {
       int i = batch->gpu_next_buf_id;
       if (batch->buf_state[i] != BUF_STATE_GPU_READY) return -1;
       return i;
}

static void 
onvm_framework_thread_init(int thread_id)
{
	int i = 0;
       
	printf("------------------start get thread %d -------------------------\n",thread_id);
	nfv_batch_t *batch = &(batch_set[thread_id]);
	/* The main thread set twice, elegant plan? */
	printf("------------------get thread %d success------------------------\n",thread_id);
	pthread_setspecific(my_batch, (void *)batch);
	batch->thread_id = thread_id;

	/* the last 1 is used to mark the allocation for not the first thread */
	if (thread_id != 0&&nf_handle_tag==GPU_NF) {
		gcudaAllocSize(0, 0, 1);
	}
	
	printf("------------------start init thread %d batch info---------------\n",thread_id);

	for (i = 0; i < NUM_BATCH_BUF; i++) {
		//在cpu模块下，用户层数据包就直接是两个uint8指针
		if(nf_handle_tag==GPU_NF)
		{	
			batch->user_bufs[i] = INIT_FUNC();
			batch->gpu_buf_id = -1;
			batch->gpu_next_buf_id = 0;
		}
		batch->pkt_ptr[i] = (struct rte_mbuf **)malloc(sizeof(struct rte_mbuf *) * MAX_BATCH_SIZE);
		//batch->gpu_buf_id = -1;
		//batch->gpu_next_buf_id = 0;
	}
}

static int 
cpu_thread(void *arg)
{
	struct thread_arg *my_arg = (struct thread_arg *)arg;
	unsigned cur_lcore = rte_lcore_id();

	if(my_arg==NULL)
		printf("-----------------------------my_arg is nullptr-----------------------\n");

	onvm_framework_thread_init(my_arg->thread_id);
	printf("------------------thread batch data init completed---------------------\n");

	if(nf_handle_tag==GPU_NF){
		pthread_mutex_lock(&lock);
		gpu_info->thread_num++;
		pthread_mutex_unlock(&lock);			
		RTE_LOG(INFO, APP, "New CPU thread %d is spawned, running on lcore %u, total_thread %d\n", my_arg->thread_id, cur_lcore, gpu_info->thread_num);
	}
	else
		RTE_LOG(INFO, APP, "New CPU thread %d is spawned, running on lcore %u\n", my_arg->thread_id, cur_lcore);

	/*
	pthread_mutex_lock(&lock);
	gpu_info->thread_num++;
	pthread_mutex_unlock(&lock);
	*/
	
	if(nf_handle_tag==GPU_NF)
		onvm_nflib_run(&(onvm_framework_cpu), my_arg->thread_id);
	else
		onvm_nflib_run(&(onvm_framework_cpu_only),my_arg->thread_id);
		
	
	RTE_LOG(INFO, APP, "Thread %d terminated on core %d\n", my_arg->thread_id, cur_lcore);

	return 0;
}



static void 
onvm_framework_spawn_thread(int thread_id, unsigned core_id)
{
	struct thread_arg *arg = (struct thread_arg *)malloc(sizeof(struct thread_arg));
	arg->thread_id = thread_id;
        printf("--------------------start nf thread %d---------------------\n",thread_id);
	if (rte_eal_remote_launch(cpu_thread, (void *)arg, core_id) == -EBUSY) {
		rte_exit(EXIT_FAILURE, "Core %d is busy, cannot allocate to run threads\n", core_id);
	}
}

static int onvm_framework_cpu_only(int thread_id){
	printf("--------------------start cpu only thread----------------------\n");
	int i, j;
	int buf_id = 0;//这个buf_id会在主循环中迭代，每个cpu最多能跑的线程个数就是其大小
	struct rte_ring *rx_q, *tx_q;
	nfv_batch_t *batch;
	int cur_buf_size;
	uint64_t starve_rx_counter = 0;
	struct timespec start, end;
	double diff;
	current_rx_qid = thread_id;

	// rx_q = cl->rx_q_new;
	while (keep_running) {
		batch = &batch_set[thread_id];
		cur_buf_size = batch->buf_size[buf_id];

		//得到自系统启动后的时间，单位是微妙
		clock_gettime(CLOCK_MONOTONIC, &start);

		// handle dropped packets
		// 下面这个操纵是在丢弃数据的同时，使得不丢弃的数据变得连续都在数组前面，要丢弃的则都在数组后面
		// 这里先进行合并，将连续数据转发后，再将丢弃的数据进行内存释放
		for (i = j = 0; i < cur_buf_size; i++) {
			struct onvm_pkt_meta *meta = onvm_get_pkt_meta(batch->pkt_ptr[buf_id][i]);
			if (meta->action != ONVM_NF_ACTION_DROP) {
				// swap
				struct rte_mbuf *p = batch->pkt_ptr[buf_id][i];
				batch->pkt_ptr[buf_id][i] = batch->pkt_ptr[buf_id][j];
				batch->pkt_ptr[buf_id][j++] = p;
			}
		}
		int num_packets = j;

		// tx
		// while (keep_running && send_token != thread_id);
		// tx_q = *(struct rte_ring * const volatile*)&cl->tx_q_new;
		if (cl->tx_qs == NULL)
			tx_q = NULL;
		else
			tx_q = *(struct rte_ring * const volatile*)&cl->tx_qs[batch->queue_id];
		int sent_packets = 0;

		
		if (likely(tx_q != NULL && num_packets != 0)) {
			//rte_ring_enqueue生产者函数，用这个函数来转发？
			// sent_packets = rte_ring_enqueue_burst(tx_q, (void **)batch->pkt_ptr[buf_id], num_packets, NULL);
			sent_packets = rte_ring_enqueue_bulk(tx_q, (void **)batch->pkt_ptr[buf_id], num_packets, NULL);
		}

		// send_token = (send_token + 1) % (gpu_info->thread_num);
		//相当于把发送位置后的值都丢弃，具体操作就是释放内存
		if (sent_packets < cur_buf_size) {
			onvm_pkt_drop_batch(batch->pkt_ptr[buf_id] + sent_packets, cur_buf_size - sent_packets);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0;

		//自旋锁
		rte_spinlock_lock(&cl->stats.update_lock);
		cl->stats.tx += sent_packets;
		cl->stats.tx_drop += num_packets - sent_packets;
		cl->stats.act_drop += cur_buf_size - num_packets;
		cl->stats.cpu_time += diff;
		rte_spinlock_unlock(&cl->stats.update_lock);

		clock_gettime(CLOCK_MONOTONIC, &start);

		// rx 尽最大能力接受数据到接受不到数据了为止
		// while (keep_running && recv_token != thread_id);
		assert(cl->worker_scale_finished <= ONVM_NUM_NF_QUEUES);
		do {
			current_rx_qid = current_rx_qid + cl->worker_scale_finished;
			if (current_rx_qid >= ONVM_NUM_NF_QUEUES) {
				current_rx_qid = thread_id;
			}
			rx_q = cl->rx_qs[current_rx_qid];
			
			//会根据实际接受到的数据包去修改batch_size
			if (BATCH_SIZE != (int)cl->batch_size) {
				BATCH_SIZE = (int)cl->batch_size;
				RTE_LOG(INFO, APP, "Batch size changed to %d\n", BATCH_SIZE);
			}

			// num_packets = rte_ring_dequeue_bulk(rx_q, (void **)batch->pkt_ptr[buf_id], BATCH_SIZE, NULL);
			num_packets = rte_ring_dequeue_burst(rx_q, (void **)batch->pkt_ptr[buf_id], BATCH_SIZE, NULL);
			if (num_packets == 0) {
				starve_rx_counter++;
				if (starve_rx_counter == STARVE_THRESHOLD) {
					buf_id = (buf_id + 1) % NUM_BATCH_BUF;
					RTE_LOG(INFO, APP, "Rx starving at thread %d\n", thread_id);
				}
			}
		} while (num_packets == 0 && keep_running);
		
		if(!keep_running)
			break;

		// recv_token = (recv_token + 1) % (gpu_info->thread_num);
		starve_rx_counter = 0;
		cur_buf_size = num_packets;
		batch->buf_size[buf_id] = cur_buf_size;

		// clock_gettime(CLOCK_MONOTONIC, &start);

		// pre-processing // todo: pass param i insteadof modify the struct
		//仅仅在预处理直接处理pkt数据包
		uint64_t rx_datalen_sample = 0;
		for (i = 0; i < cur_buf_size; i++) 
		{
			//batch_handle
			CPU_BATCH_FUNC(batch->pkt_ptr[buf_id][i]);
		}
		rx_datalen_sample = cur_buf_size > 0 ? cur_buf_size * batch->pkt_ptr[buf_id][0]->data_len : 0;

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0;

		rte_spinlock_lock(&cl->stats.update_lock);
		cl->stats.rx += cur_buf_size;		
		cl->stats.rx_datalen += rx_datalen_sample;
		cl->stats.cpu_time += diff;
		rte_spinlock_unlock(&cl->stats.update_lock);

		// launch kernel
		// 这里相当于判断其接受数据是否大于0，大于0，那么接下来可以进入gpu模式操作
		if (cur_buf_size > 0) {
			buf_id = (buf_id + 1) % NUM_BATCH_BUF;
		}
	}

	return 0;
}

//cpu处理模块
static int
onvm_framework_cpu(int thread_id)
{
	int i, j;
	int buf_id = 0;//这个buf_id会在主循环中迭代，每个cpu最多能跑的线程个数就是其大小
	struct rte_ring *rx_q, *tx_q;
	nfv_batch_t *batch;
	int cur_buf_size;
	uint64_t starve_rx_counter = 0;
	uint64_t starve_gpu_counter = 0;
	struct timespec start, end;
	double diff;
	current_rx_qid = thread_id;

	// rx_q = cl->rx_q_new;
	while (keep_running) {
		batch = &batch_set[thread_id];
		//gpu饿了是什么意思，必须当前cpu_ready后才会进行接下来的操作
        if (batch->buf_state[buf_id] != BUF_STATE_CPU_READY) {
			starve_gpu_counter++;
			if (starve_gpu_counter == STARVE_THRESHOLD) {
				//当前项没有处理完时，选择循环选择下一个缓冲区
				buf_id = (buf_id + 1) % NUM_BATCH_BUF;
				RTE_LOG(INFO, APP, "GPU starving\n");
			}
			continue;
		}
		starve_gpu_counter = 0;
		cur_buf_size = batch->buf_size[buf_id];

		clock_gettime(CLOCK_MONOTONIC, &start);

		// post-processing
		// 这里所谓的后处理是将用户数据传递给rte_mem why?
		for (i = 0; i < cur_buf_size; i++) {
			POST_FUNC(batch->user_bufs[buf_id], batch->pkt_ptr[buf_id][i], i);
		}

		// handle dropped packets
		// 下面这个操纵是在丢弃数据的同时，使得不丢弃的数据变得连续都在数组前面，要丢弃的则都在数组后面
		// 这里先进行合并，将连续数据转发后，再将丢弃的数据进行内存释放
		for (i = j = 0; i < cur_buf_size; i++) {
			struct onvm_pkt_meta *meta = onvm_get_pkt_meta(batch->pkt_ptr[buf_id][i]);
			if (meta->action != ONVM_NF_ACTION_DROP) {
				// swap
				struct rte_mbuf *p = batch->pkt_ptr[buf_id][i];
				batch->pkt_ptr[buf_id][i] = batch->pkt_ptr[buf_id][j];
				batch->pkt_ptr[buf_id][j++] = p;
			}
		}
		int num_packets = j;

		// tx
		// while (keep_running && send_token != thread_id);
		// tx_q = *(struct rte_ring * const volatile*)&cl->tx_q_new;
		if (cl->tx_qs == NULL)
			tx_q = NULL;
		else
			tx_q = *(struct rte_ring * const volatile*)&cl->tx_qs[batch->queue_id];
		int sent_packets = 0;

		
		if (likely(tx_q != NULL && num_packets != 0)) {
			//rte_ring_enqueue生产者函数，用这个函数来转发？
			// sent_packets = rte_ring_enqueue_burst(tx_q, (void **)batch->pkt_ptr[buf_id], num_packets, NULL);
			sent_packets = rte_ring_enqueue_bulk(tx_q, (void **)batch->pkt_ptr[buf_id], num_packets, NULL);
		}

		// send_token = (send_token + 1) % (gpu_info->thread_num);
		//相当于把发送位置后的值都丢弃，具体操作就是释放内存
		
		//为什么不开个循环多转发几次把数据包都转发掉
		if (sent_packets < cur_buf_size) {
			onvm_pkt_drop_batch(batch->pkt_ptr[buf_id] + sent_packets, cur_buf_size - sent_packets);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0;

		//自旋锁
		rte_spinlock_lock(&cl->stats.update_lock);
		cl->stats.tx += sent_packets;
		cl->stats.tx_drop += num_packets - sent_packets;
		cl->stats.act_drop += cur_buf_size - num_packets;
		cl->stats.cpu_time += diff;
		rte_spinlock_unlock(&cl->stats.update_lock);

		clock_gettime(CLOCK_MONOTONIC, &start);

		// rx 尽最大能力接受数据到接受不到数据了为止
		// while (keep_running && recv_token != thread_id);
		// 每个nf最多两个接收队列
		assert(cl->worker_scale_finished <= ONVM_NUM_NF_QUEUES);
		do {
			current_rx_qid = current_rx_qid + cl->worker_scale_finished;
			if (current_rx_qid >= ONVM_NUM_NF_QUEUES) {
				current_rx_qid = thread_id;
			}
			rx_q = cl->rx_qs[current_rx_qid];
			
			//会根据实际接受到的数据包去修改batch_size
			if (BATCH_SIZE != (int)cl->batch_size) {
				BATCH_SIZE = (int)cl->batch_size;
				RTE_LOG(INFO, APP, "Batch size changed to %d\n", BATCH_SIZE);
			}

			// num_packets = rte_ring_dequeue_bulk(rx_q, (void **)batch->pkt_ptr[buf_id], BATCH_SIZE, NULL);
			num_packets = rte_ring_dequeue_burst(rx_q, (void **)batch->pkt_ptr[buf_id], BATCH_SIZE, NULL);
			if (num_packets == 0) {
				starve_rx_counter++;
				if (starve_rx_counter == STARVE_THRESHOLD) {
					//buf_id = (buf_id + 1) % NUM_BATCH_BUF;
					RTE_LOG(INFO, APP, "Rx starving at thread %d\n", thread_id);
				}
			}
		} while (num_packets == 0);
		
#ifdef PRINT_ACTUAL_BATCH_SIZE
		static int batch_freq = 0;
		if (batch_freq % 10000 == 0) {
			printf("Actual batch size: %d\n", num_packets);
			batch_freq = 0;
		}
		batch_freq++;
#endif

#ifdef RING_QUEUING_LATENCY
		for (i = 0; i < num_packets; i++) {
			if (unlikely(batch->pkt_ptr[buf_id][i]->seqn == LATENCY_MAGIC)) {
				struct timespec prev;
				// prev.tv_sec = (uint32_t) batch->pkt_ptr[buf_id][i]->timestamp;
				// prev.tv_usec = batch->pkt_ptr[buf_id][i]->udata64;
				prev.tv_sec = batch->pkt_ptr[buf_id][i]->tv_sec;
				prev.tv_nsec = batch->pkt_ptr[buf_id][i]->tv_nsec;
				double timediff_usec = time_diff(prev);
                printf("Ring latency: %.3f ms\n", timediff_usec / 1e3);

				clock_gettime(CLOCK_MONOTONIC, &prev);
				batch->pkt_ptr[buf_id][i]->tv_sec = prev.tv_sec;
				batch->pkt_ptr[buf_id][i]->tv_nsec = prev.tv_nsec;
				break;
			}
		}
#endif
		// recv_token = (recv_token + 1) % (gpu_info->thread_num);
		starve_rx_counter = 0;
		cur_buf_size = num_packets;
		batch->buf_size[buf_id] = cur_buf_size;

		// clock_gettime(CLOCK_MONOTONIC, &start);

		// pre-processing // todo: pass param i insteadof modify the struct
		uint64_t rx_datalen_sample = 0;
		for (i = 0; i < cur_buf_size; i++) {
#ifdef ENABLE_PSTACK
			pstack_process((char *)onvm_pkt_ipv4_hdr(batch->pkt_ptr[buf_id][i]), batch->pkt_ptr[buf_id][i]->data_len - sizeof(struct ether_hdr), thread_id);
#endif		
			//将接受到的数据传递给user_bufs
			PRE_FUNC(batch->user_bufs[buf_id], batch->pkt_ptr[buf_id][i], i);
		}
		rx_datalen_sample = cur_buf_size > 0 ? cur_buf_size * batch->pkt_ptr[buf_id][0]->data_len : 0;

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0;

		rte_spinlock_lock(&cl->stats.update_lock);
		cl->stats.rx += cur_buf_size;		
		cl->stats.rx_datalen += rx_datalen_sample;
		cl->stats.cpu_time += diff;
		rte_spinlock_unlock(&cl->stats.update_lock);

		// launch kernel
		// 这里相当于判断其接受数据是否大于0，大于0，那么接下来可以进入gpu模式操作
		if (cur_buf_size > 0) {
			batch->buf_state[buf_id] = BUF_STATE_GPU_READY;
            buf_id = (buf_id + 1) % NUM_BATCH_BUF;
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

	// recv_token = send_token = 0;
#ifdef ENABLE_PSTACK
	pstack_info.ip_thread_local = rte_malloc(PSTACK_IP_INFO_NAME, MAX_CPU_THREAD_NUM * PSTACK_IP_INFO_SIZE, 0);
	pstack_info.tcp_thread_local = rte_malloc(PSTACK_TCP_INFO_NAME, MAX_CPU_THREAD_NUM * PSTACK_TCP_INFO_SIZE, 0);

	pstack_init(pstack_info, MAX_CPU_THREAD_NUM);
#endif
}


void 
onvm_framework_cpu_only_wait(void){
	signal(SIGINT,onvm_nflib_handle_signal);
	while(keep_running);
	onvm_nflib_stop();
}

void
onvm_framework_start_cpu(init_func_t user_init_buf_func, pre_func_t user_pre_func, post_func_t user_post_func,cpu_batch_handle cpu_batch_func,int handle_tag)
{
	printf("-------------------start load cpu-----------\n");
	nf_handle_tag= handle_tag;

	if(nf_handle_tag==GPU_NF)
	{
		INIT_FUNC = user_init_buf_func;
		PRE_FUNC = user_pre_func;
		POST_FUNC = user_post_func;
	}
	else
		CPU_BATCH_FUNC=cpu_batch_func;

	printf("-------------------finsh load cpu/gpu function---------------\n");
	
	int i;
    unsigned cur_lcore = rte_lcore_id();
	for (i = 0; i < INIT_WORKER_THREAD_NUM; i ++) {
		/* Better to wait for a while between launching two threads, don't know why */
		sleep(1);
        	cur_lcore =rte_get_next_lcore(cur_lcore, 1, 1);
		cl->worker_scale_finished++;
		onvm_framework_spawn_thread(i, cur_lcore);
		last_assigned_core_id = cur_lcore;
	}
}

void
onvm_framework_start_gpu(gpu_htod_t user_gpu_htod, gpu_dtoh_t user_gpu_dtoh, gpu_set_arg_t user_gpu_set_arg)
{
	int gpu_buf_id;
	int batch_id;
	unsigned int i;
	nfv_batch_t *batch;

	/* Listen for ^C and docker stop so we can exit gracefully */
	signal(SIGINT, onvm_nflib_handle_signal);

	if (user_gpu_set_arg == NULL || user_gpu_htod == NULL || user_gpu_dtoh == NULL) {
		rte_exit(EXIT_FAILURE, "GPU function is NULL\n");
	}

	//通过自旋来确保cpu已经完成了cpu_thread的执行，开始进入onvm_framework_cpu
	while (gpu_info->thread_num != (unsigned int)INIT_WORKER_THREAD_NUM && keep_running) ;

	unsigned cur_lcore = rte_lcore_id();
	RTE_LOG(INFO, APP, "GPU thread is running on lcore %u\n", cur_lcore);
	printf("[Press Ctrl-C to quit ...]\n\n");

	for (; keep_running ;) {
		/* 1. Wait until the batch size is reached for the first thread.
		 * We have load balance among all threads, so their batch size are the same. */
		RTE_LOG(DEBUG, APP, "GPU thread is launching kernel\n");

		//处理d2h
		//在单rx线程的情况下 gpu_info->thread_num 也等于1
		for (i = 0; i < gpu_info->thread_num; i++) {
			batch = &batch_set[i];
			//试图获得该gpu线程对应的数据包
			//如果获得不到响应，并且之前已经处理过数据包的话就退出循环
			//在流回调函数执行后，才会调用下面的模块
			if (batch->gpu_buf_id != -1 && gcudaPollForStreamSyncResponse(i)) {
				//gpu_state指得应该是gpu_ready
				if (batch->gpu_state == 1) {
					//将gpu数据拷贝给用户缓冲区
					user_gpu_dtoh(batch->user_bufs[batch->gpu_buf_id], batch->buf_size[batch->gpu_buf_id], i);
					//gpu流同步请求的申请和加入队列
					gcudaStreamSynchronize(i);

					//自旋锁更新缓冲区数据
					rte_spinlock_lock(&cl->stats.update_lock);
					cl->stats.batch_size += batch->buf_size[batch->gpu_buf_id];
					cl->stats.batch_cnt++;
					rte_spinlock_unlock(&cl->stats.update_lock);
					batch->gpu_state = 0;
				} else {
					gpu_buf_id = batch->gpu_buf_id;
					batch->gpu_buf_id = -1;
					batch->buf_state[gpu_buf_id] = BUF_STATE_CPU_READY;
				}
			}
		}

		//处理h2d
		//具体的执行细节不了解
		gpu_buf_id = -1;
		for (i = 0; gpu_buf_id == -1 && i < gpu_info->thread_num; i++) {
			batch = &batch_set[i];
			if (batch->gpu_buf_id != -1) continue;
			batch_id = i;
			//在gpu ready也就是数据包已经从pkt复制到user_buf的情况下，获得数据包id
			gpu_buf_id = gpu_get_batch(batch);
		}

		//判断是否继续运行和是否获得了正确的数据包
		if (!keep_running) break;
		if (gpu_buf_id == -1) continue;
		batch->gpu_buf_id = gpu_buf_id;
		batch->gpu_next_buf_id = (gpu_buf_id + 1) % NUM_BATCH_BUF;

		// Automatic scaling
		// 目标cpu需要数量比当前执行数量多，那么uhi通过spawn_thread去开辟更多的cpu线程
		if (cl->worker_scale_target > cl->worker_scale_finished) {
			RTE_LOG(INFO, APP, "CPU Scaling: %u -> %u\n", cl->worker_scale_finished, cl->worker_scale_target);
			int num = cl->worker_scale_target - cl->worker_scale_finished;
			int base = cl->worker_scale_finished;
			unsigned lcore = last_assigned_core_id;
			for (int i = base; i < num + base; i++) {
				// sleep(1);
        		lcore =	rte_get_next_lcore(lcore, 1, 1);
				rte_eal_wait_lcore(lcore);
				onvm_framework_spawn_thread(i, lcore);
				last_assigned_core_id = lcore;
				cl->worker_scale_finished++;
			}
		}

		/* 3. Launch kernel - USER DEFINED */
		gcudaRecordStart(batch_id);
		user_gpu_htod(batch->user_bufs[gpu_buf_id], batch->buf_size[gpu_buf_id], batch_id);
		user_gpu_set_arg(batch->user_bufs[gpu_buf_id], gpu_info->args[batch_id], gpu_info->arg_info[batch_id], batch->buf_size[gpu_buf_id]);
		gcudaLaunchKernel(batch_id);
		batch->gpu_state = 1;
	}

	onvm_nflib_stop(); // clean up
}
/* ======================================= */

void onvm_framework_get_hint(uint8_t* h2d_hint,uint8_t* d2h_hint , uint16_t* h2d_offset , uint16_t* d2h_offset,packet_sync_global_t* sync_data)
{
	sync_data->payload_size = -1;
	
	*d2h_hint=(uint8_t)((sync_plan & 0xFF)>>3);
	sync_data->d2h_payload_flag=((*d2h_hint & 1) == 1); 
	*h2d_hint=(uint8_t)(((sync_plan >> 8) & 0xFF)>>3);
	sync_data->h2d_payload_flag=((*h2d_hint & 1) == 1); 

	int index = 0;
	int sync_num = 0;
	uint8_t tp_hint = (*h2d_hint);
	while(tp_hint != 0)
	{
		index ++;
		if((tp_hint & 1) == 1)
		{
			sync_num++;
			switch (SYNC_DATA_COUNT - index)
			{
				case SYNC_SOURCE_IP:
					h2d_offset[sync_num-1] = 0;
					break;
				case SYNC_DEST_IP:
					h2d_offset[sync_num-1] = 4;
					break;
				case SYNC_SOURCE_PORT:
					h2d_offset[sync_num-1] = 8;
					break;
				case SYNC_DEST_PORT:
					h2d_offset[sync_num-1] = 10;		
					break;
				case SYNC_TCP_FLAGS:
					h2d_offset[sync_num-1] = 12;
					break;
			}
		}
		tp_hint >>= 1;
	}
	sync_data->h2d_sync_num = sync_num;

	sync_num = 0;
	index = 0;
	tp_hint = (*d2h_hint);
	while(tp_hint != 0)
	{
		index ++;
		if((tp_hint & 1) == 1)
		{
			sync_num++;
			switch (SYNC_DATA_COUNT - index)
			{
				case SYNC_SOURCE_IP:
					d2h_offset[sync_num-1] = 0;
					break;
				case SYNC_DEST_IP:
					d2h_offset[sync_num-1] = 4;
					break;
				case SYNC_SOURCE_PORT:
					d2h_offset[sync_num-1] = 8;
					break;
				case SYNC_DEST_PORT:
					d2h_offset[sync_num-1] = 10;		
					break;
				case SYNC_TCP_FLAGS:
					d2h_offset[sync_num-1] = 12;
					break;
			}
		}
		tp_hint >>= 1;
	}
	sync_data->d2h_sync_num = sync_num;

	printf("EAL: h2d sync number:%d d2h sync number:%d\n",sync_data->h2d_sync_num,sync_data->d2h_sync_num);
	printf("EAL: H2D offset:");
	for(int i=0;i<(sync_data->h2d_sync_num);i++)
		printf("%d ",h2d_offset[i]);
	printf("\nEAL: D2H offset:");
	for(int i=0;i<(sync_data->d2h_sync_num);i++)
		printf("%d ",d2h_offset[i]);
	printf("\n");
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

	//给host内存分配dpdk层空间
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
gcudaMemcpyHtoD(CUdeviceptr dst, void *src, int size, int sync, int thread_id)
{
	UNUSED(sync);
	nfv_batch_t *batch = &(batch_set[thread_id]);
	assert(batch->thread_id == thread_id);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_MEMCPY_HTOD_ASYNC;
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
}

void
gcudaMemcpyDtoH(void *dst, CUdeviceptr src, int size, int sync, int thread_id)
{
	UNUSED(sync);
	nfv_batch_t *batch = &(batch_set[thread_id]);
	assert(batch->thread_id == thread_id);

	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get resquest memory\n");
	}

	req->type = REQ_GPU_MEMCPY_DTOH_ASYNC;
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

static void
gcudaRecordStart(int thread_id)
{
	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_GPU_RECORD_START;
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

//从请求内存池中申请内存，再将请求压入请求环形队列和内存池，这里的请求是gpu同步流请求
static void
gcudaStreamSynchronize(int thread_id)
{
	struct nf_req *req;
	if (rte_mempool_get(nf_request_mp, (void **)&req) < 0) {
		rte_exit(EXIT_FAILURE, "Failed to get request memory\n");
	}

	req->type = REQ_GPU_SYNC_STREAM;
	req->instance_id = nf_info->instance_id;
	req->thread_id = thread_id;

	RTE_LOG(DEBUG, APP, "[G] cudaStreamSync\n");

	if (rte_ring_enqueue(nf_request_queue, req) < 0) {
		rte_mempool_put(nf_request_mp, req);
		rte_exit(EXIT_FAILURE, "Cannot send request_info to scheduler\n");
	}
}

//从响应的环形缓冲区中获取数据，并将响应放入到dpdk的nf_response_mp中
static int
gcudaPollForStreamSyncResponse(int thread_id)
{
	struct rte_ring *nf_response_ring = cl->response_q[thread_id];

	struct nf_rsp *rsp;
	if (rte_ring_dequeue(nf_response_ring, (void **)&rsp) != 0)
		return 0;

	if (rsp->type != RSP_GPU_SYNC_STREAM)
		rte_exit(EXIT_FAILURE, "Wrong response type %d\n", rsp->type);

	rte_mempool_put(nf_response_mp, rsp);

	return 1;
}
