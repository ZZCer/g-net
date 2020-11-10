#ifndef _ONVM_FRAMEWORK_H_
#define _ONVM_FRAMEWORK_H_

#include <cuda.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>
#include <rte_config.h>
#include "onvm_common.h"

#define MAX_PKT_LEN 1514

#define SYNC 0
#define ASYNC 1

//表示网络功能的计算是在CPU计算还是在GPU计算
#define CPU_NF 0
#define GPU_NF 1

/* Each CPU worker holds such a data structure */
typedef struct nfv_batch_s
{
	//下述这两个存储数据的数组有什么区别么,共享内存/用户空间之分
	void *user_bufs[NUM_BATCH_BUF];
	struct rte_mbuf **pkt_ptr[NUM_BATCH_BUF];

	int buf_size[NUM_BATCH_BUF];
	volatile int buf_state[NUM_BATCH_BUF];

	int thread_id;
	volatile int gpu_buf_id;
	int gpu_next_buf_id;
	int gpu_state;

	int queue_id;

	void *host_mem_addr_base;
	void *host_mem_addr_cur;
	int host_mem_size_total;
	int host_mem_size_left;
} nfv_batch_t;

enum {
	BUF_STATE_CPU_READY = 0,
	BUF_STATE_GPU_READY,
};

typedef struct context_s{
	int thread_id;
} context_t;

typedef void *(*init_func_t)(void);
typedef void (*cpu_batch_handle)(struct rte_mbuf*);
typedef void (*pre_func_t)(void *, struct rte_mbuf *, int);
typedef void (*post_func_t)(void *, struct rte_mbuf *, int);
typedef void (*gpu_htod_t)(void *, int, unsigned int);
typedef void (*gpu_dtoh_t)(void *, int, unsigned int);
typedef void (*gpu_set_arg_t)(void *, void *, void *, int);

void onvm_framework_start_cpu(init_func_t, pre_func_t, post_func_t,cpu_batch_handle,int nf_handle_tag);
void onvm_framework_start_gpu(gpu_htod_t, gpu_dtoh_t, gpu_set_arg_t);

void onvm_framework_install_kernel_perf_parameter(double k1, double b1, double k2, double b2);
void onvm_framework_install_kernel_perf_para_set(double *u_k1, double *u_b1, double *u_k2, double *u_b2, unsigned int *pkt_size, unsigned int *line_start_batch, unsigned int para_num);
void onvm_framework_init(const char *module_file, const char *kernel_name);

void gcudaAllocSize(int size1, int size2, int first);
void gcudaMalloc(CUdeviceptr *p, int size);
void gcudaHostAlloc(void **p, int size);
void gcudaMemcpyHtoD(CUdeviceptr dst, void *src, int size, int sync, int thread_id);
void gcudaMemcpyDtoH(void *dst, CUdeviceptr src, int size, int sync, int thread_id);
void gcudaLaunchKernel(int thread_id);
void gcudaLaunchKernel_allStream(void);
void gcudaDeviceSynchronize(void);

void onvm_framework_cpu_only_wait(void);
#endif
