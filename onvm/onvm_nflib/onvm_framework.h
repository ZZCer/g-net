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

/* Each CPU worker holds such a data structure */
typedef struct nfv_batch_s
{
	void *user_buf;
	struct rte_mbuf **pkt_ptr;

	int buf_size;
	volatile int buf_state;
	rte_spinlock_t processing_lock;

	int stream_id;
} nfv_batch_t;

enum {
	BUF_STATE_CPU_READY = 0,
	BUF_STATE_GPU_READY,
};

typedef struct context_s{
	int thread_id;
} context_t;

typedef void *(*init_func_t)(void);
typedef void (*pre_func_t)(void *, struct rte_mbuf *, int);
typedef void (*post_func_t)(void *, struct rte_mbuf *, int);
typedef void (*gpu_htod_t)(void *, int);
typedef void (*gpu_dtoh_t)(void *, int);
typedef void (*gpu_set_arg_t)(void *, void *, void *, int);

void onvm_framework_start_cpu(init_func_t, pre_func_t, post_func_t);
void onvm_framework_start_gpu(gpu_htod_t, gpu_dtoh_t, gpu_set_arg_t);

void onvm_framework_install_kernel_perf_parameter(double k1, double b1, double k2, double b2);
void onvm_framework_install_kernel_perf_para_set(double *u_k1, double *u_b1, double *u_k2, double *u_b2, unsigned int *pkt_size, unsigned int *line_start_batch, unsigned int para_num);
void onvm_framework_init(const char *module_file, const char *kernel_name);

void gcudaAllocSize(int size1, int size2);
void gcudaMalloc(CUdeviceptr *p, int size);
void gcudaHostAlloc(void **p, int size);
void gcudaMemcpyHtoD(CUdeviceptr dst, void *src, int size);
void gcudaMemcpyDtoH(void *dst, CUdeviceptr src, int size);
void gcudaLaunchKernel(void);
void gcudaLaunchKernel_allStream(void);
void gcudaDeviceSynchronize(void);

#endif
