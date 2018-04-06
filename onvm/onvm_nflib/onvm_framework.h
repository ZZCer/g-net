#ifndef _ONVM_FRAMEWORK_H_
#define _ONVM_FRAMEWORK_H_

#include <cuda.h>
#include "onvm_common.h"

#define MAX_PKT_LEN 1514

#define SYNC 0
#define ASYNC 1

/* Each CPU worker holds such a data structure */
typedef struct nfv_batch_s
{
	void *user_bufs[NUM_BATCH_BUF];
	struct rte_mbuf **pkt_ptr[NUM_BATCH_BUF];

	volatile int receiver_buf_id;
	volatile int sender_buf_id;
	volatile int available_buf_id[NUM_BATCH_BUF];
	int gpu_buf_id;

	unsigned int thread_id;
	int64_t post_idx;

	pthread_mutex_t mutex_sender_buf_id; 
	pthread_mutex_t mutex_available_buf_id; 
	pthread_mutex_t mutex_batch_launch; 

	void *host_mem_addr_base;
	void *host_mem_addr_cur;
	int host_mem_size_total;
	int host_mem_size_left;
} nfv_batch_t;

typedef struct context_s{
	int thread_id;
} context_t;

struct gpu_packet_s {
	struct ipv4_hdr ipv4_hdr_data;
	union {
		struct tcp_hdr tcp_hdr_data;
		struct udp_hdr udp_hdr_data;
	};
	uint8_t payload[MAX_PKT_LEN];
} __attribute__((aligned(16)));

void onvm_framework_start_cpu(void *(*user_init_buf_func)(void), 
						void (*user_batch_func)(void *,  struct rte_mbuf *),
						void (*user_post_func)(void *, struct rte_mbuf *, int));
void onvm_framework_start_gpu(void (*user_gpu_htod)(void *, unsigned int),
						void (*user_gpu_dtoh)(void *, unsigned int),
						void (*user_gpu_set_arg)(void *, void *, void *));

void onvm_framework_install_kernel_perf_parameter(double k1, double b1, double k2, double b2);
void onvm_framework_install_kernel_perf_para_set(double *u_k1, double *u_b1, double *u_k2, double *u_b2, unsigned int *pkt_size, unsigned int *line_start_batch, unsigned int para_num);
void onvm_framework_init(const char *module_file, const char *kernel_name);

void gcudaAllocSize(int size1, int size2, int first);
void gcudaMalloc(CUdeviceptr *p, int size);
void gcudaHostAlloc(void **p, int size);
void gcudaMemcpyHtoD(CUdeviceptr dst, void *src, int size, int sync, unsigned int thread_id);
void gcudaMemcpyDtoH(void *dst, CUdeviceptr src, int size, int sync, unsigned int thread_id);
void gcudaLaunchKernel(int thread_id);
void gcudaLaunchKernel_allStream(void);
void gcudaDeviceSynchronize(void);

#endif
