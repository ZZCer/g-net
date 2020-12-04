#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_ether.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_framework.h"

#define NF_TAG "GPU"
#define MAX_SIZE_PORTS 65535
#define SIZE_IP 4 //下述单位是字节
#define SIZE_PORT 2

//从srcIP开始，用来记录需要同步的数据的offset,同时有一个sync_num用来记录需要同步的数据个数
uint16_t *h2d_offset;
uint16_t *d2h_offset;
uint16_t h2d_sync_num;
uint16_t d2h_sync_num;

//上述这些用于同步的信息需要作为全局变量传递给gpu
static CUdeviceptr h2d_offset_d;
static CUdeviceptr d2h_offset_d;

//h2d hint和d2h hint,由于不是指针，因此可以直接通过地址将数据传递给gpu
static uint8_t h2d_hint;
static uint8_t d2h_hint; 

typedef struct my_buf_s {
	/* Stores real data */
	CUdeviceptr *host_in;
	uint8_t *host_out;

	CUdeviceptr device_in;
	CUdeviceptr device_out;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));

	gcudaHostAlloc((void **)&(buf->host_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaHostAlloc((void **)&(buf->host_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	
	gcudaMalloc(&(buf->device_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaMalloc(&(buf->device_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	
	return buf;
}

static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	buf->host_in[pkt_idx] = onvm_pkt_gpu_ptr(pkt);
}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;
}

static void user_gpu_htod(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->device_in, buf->host_in, job_num * sizeof(CUdeviceptr), ASYNC, thread_id);
}

static void user_gpu_dtoh(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_out, buf->device_out, job_num * sizeof(uint8_t), ASYNC, thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	uint64_t arg_num = 4;
	uint64_t offset = 0;

	info[0] = arg_num;

	info[1] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->device_in), sizeof(buf->device_in));
	offset += sizeof(buf->device_in);

	info[2] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(job_num), sizeof(job_num));
	offset += sizeof(job_num);

    info[3] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->device_out), sizeof(buf->device_out));
	offset += sizeof(buf->device_out);
}

static void init_main(void)
{
	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * sizeof(CUdeviceptr) + MAX_BATCH_SIZE * sizeof(uint8_t),0,0);                            
}

static void init_gpu_schedule(void)
{
	/* Initialize the GPU info, onvm_framework_init should be performed before onvm_nflib_init */
	const char *module_file = "../GPU_test/gpu/gpu_test.ptx";
	const char *kernel_name = "gpu_test";
	onvm_framework_init(module_file, kernel_name);

	double K1 = 0;
	double B1 = 0;
	double K2 = 0;
	double B2 = 0;

	onvm_framework_install_kernel_perf_parameter(K1, B1, K2, B2);
}

int main(int argc, char *argv[])
{
	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_GPU,GPU_NF,(&init_gpu_schedule))) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	// 此时这里应该只需要分配host数据就可以了
	init_main();

	//获取hint信息
	onvm_framework_get_hint(&h2d_hint , &d2h_hint , 
							h2d_offset, d2h_offset , 
							&h2d_sync_num , 
							&d2h_sync_num);
	
	/* Initialization is done, start threads */
	onvm_framework_start_cpu(&(init_host_buf), &(user_batch_func), &(user_post_func),NULL,GPU_NF);

	onvm_framework_start_gpu(&(user_gpu_htod), &(user_gpu_dtoh), &(user_gpu_set_arg));

	printf("If we reach here, program is ending\n");
	return 0;
}
