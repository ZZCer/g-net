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
#include <rte_udp.h>
#include <rte_ether.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_framework.h"

#include "gpu/rules.h"

#define NF_TAG "nids"

static CUdeviceptr dev_acGPU;

typedef struct my_buf_s {
	/* Stores real data */
	CUdeviceptr *host_pkt;
	uint16_t *host_res;
	CUdeviceptr dev_pkt;
	CUdeviceptr dev_res;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));

	gcudaHostAlloc((void **)&(buf->host_pkt), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaHostAlloc((void **)&(buf->host_res), MAX_BATCH_SIZE * sizeof(uint16_t));

	gcudaMalloc(&(buf->dev_pkt), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaMalloc(&(buf->dev_res), MAX_BATCH_SIZE * sizeof(uint16_t));

	return buf;
}

static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	buf->host_pkt[pkt_idx] = onvm_pkt_gpu_ptr(pkt);
}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	/* Do not drop packet yet, let the following NFs to process in evaluation */
	return;

	buf_t *buf = (buf_t *)cur_buf;

	/* Drops the packet if malicious content is detected */
	if (buf->host_res[pkt_idx] != 0) {
		struct onvm_pkt_meta* meta;
		meta = onvm_get_pkt_meta((struct rte_mbuf *)pkt);
		meta->action = ONVM_NF_ACTION_DROP;
		RTE_LOG(INFO, APP, "Packet is dropped by the NIDS\n");
	}
}

static void user_gpu_htod(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->dev_pkt, buf->host_pkt, job_num * sizeof(CUdeviceptr), ASYNC, thread_id);
}

static void user_gpu_dtoh(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_res, buf->dev_res, job_num * sizeof(uint16_t), ASYNC, thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	uint64_t arg_num = 5;
	uint64_t offset = 0;

	info[0] = arg_num;

	info[1] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(dev_acGPU), sizeof(dev_acGPU));
	offset += sizeof(dev_acGPU);

	info[2] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_pkt), sizeof(buf->dev_pkt));
	offset += sizeof(buf->dev_pkt);

	info[4] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_res), sizeof(buf->dev_res));
	offset += sizeof(buf->dev_res);
	
	info[5] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &job_num, sizeof(job_num));
	offset += sizeof(job_num);
}

static void init_main(void)
{
	ListRoot *listroot;
	listroot = configrules("test-rules");
	precreatearray(listroot);

	RuleSetRoot *rsr = listroot->TcpListRoot->prmGeneric->rsr;

	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * sizeof(CUdeviceptr) // input buffer
			+ MAX_BATCH_SIZE * sizeof(uint16_t), // output result
			MAX_STATE * 257 * sizeof(uint16_t),  // state table
			0);                                  // first time

	void *host_acGPU;

	gcudaMalloc(&dev_acGPU, MAX_STATE * 257 * sizeof(uint16_t));
	gcudaHostAlloc((void **)&host_acGPU, MAX_STATE * 257 * sizeof(uint16_t));

	/* copy the rule set to the shared memory first */
	rte_memcpy(host_acGPU, rsr->acGPU, MAX_STATE * 257 * sizeof(uint16_t));

	/* Transfer to GPU with the data in the shared memory */
	gcudaMemcpyHtoD(dev_acGPU, host_acGPU, MAX_STATE * 257 * sizeof(uint16_t), SYNC, 0);
}

static void init_gpu_schedule(void)
{
	/* Initialize the GPU info, onvm_framework_init should be performed before onvm_nflib_init */
	const char *module_file = "../nids/gpu/nids.ptx";
	const char *kernel_name = "match";
	onvm_framework_init(module_file, kernel_name);

	unsigned int pkt_size[6] = {64, 128, 256, 512, 1024, 1518};
	unsigned int line_start_batch[6] = {1024, 640, 550, 512, 512, 512};
	double k1[6] = {0,        0.032422, 0.093672, 0.214922, 0.486172, 0.776953};
	double b1[6] = {22.30,       39.15,    81.19,   171.08,   337.92,    489.3};
	double k2[6] = {0.007598, 0.014195, 0.033901, 0.081016, 0.169258, 0.215594};
	double b2[6] = {13.836,     19.532,   19.511,   19.021,   17.618,   32.829};

	onvm_framework_install_kernel_perf_para_set(k1, b1, k2, b2, pkt_size, line_start_batch, 6);
}

int main(int argc, char *argv[])
{
	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_NIDS, &(init_gpu_schedule))) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	init_main();

	/* Initialization is done, start threads */
	onvm_framework_start_cpu(&(init_host_buf), &(user_batch_func), &(user_post_func));

	onvm_framework_start_gpu(&(user_gpu_htod), &(user_gpu_dtoh), &(user_gpu_set_arg));

	printf("If we reach here, program is ending\n");
	return 0;
}
