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
#include <rte_byteorder.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_framework.h"

#include "gpu/crypto_size.h"

#define NF_TAG "ipsec"

typedef struct my_buf_s {
	/* Stores real data */
	uint64_t job_num;
	uint8_t *host_in;
	uint32_t *host_pkt_offset;
	uint16_t *host_length;
	uint8_t *host_aes_key;
	uint8_t *host_hmac_key;
	uint8_t *host_out;
	CUdeviceptr dev_in;
	CUdeviceptr dev_pkt_offset;
	CUdeviceptr dev_length;
	CUdeviceptr dev_aes_key;
	CUdeviceptr dev_hmac_key;
	CUdeviceptr dev_out;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));

	gcudaHostAlloc((void **)&(buf->host_in), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t));
	gcudaHostAlloc((void **)&(buf->host_pkt_offset), (MAX_BATCH_SIZE + 1) * sizeof(uint32_t));
	gcudaHostAlloc((void **)&(buf->host_length), MAX_BATCH_SIZE * PKT_LENGTH_SIZE);
	gcudaHostAlloc((void **)&(buf->host_aes_key), MAX_BATCH_SIZE * AES_KEY_SIZE);
	gcudaHostAlloc((void **)&(buf->host_hmac_key), MAX_BATCH_SIZE * HMAC_KEY_SIZE);
	gcudaHostAlloc((void **)&(buf->host_out), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t));

	gcudaMalloc(&(buf->dev_in), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t));
	gcudaMalloc(&(buf->dev_pkt_offset), (MAX_BATCH_SIZE + 1) * sizeof(uint32_t));
	gcudaMalloc(&(buf->dev_length), MAX_BATCH_SIZE * PKT_LENGTH_SIZE);
	gcudaMalloc(&(buf->dev_aes_key), MAX_BATCH_SIZE * AES_KEY_SIZE);
	gcudaMalloc(&(buf->dev_hmac_key), MAX_BATCH_SIZE * HMAC_KEY_SIZE);
	gcudaMalloc(&(buf->dev_out), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t));

	return buf;
}

static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	struct udp_hdr *udp = onvm_pkt_udp_hdr(pkt);
	if (udp == NULL) {
		rte_exit(EXIT_FAILURE, "UDP is NULL");
	}

	/* Paylod */
	uint8_t *pkt_data = (uint8_t *)udp + sizeof(struct udp_hdr);
	uint16_t payload_len = rte_be_to_cpu_16(udp->dgram_len) - 8; /* 8 byte udp header size */
	uint16_t pad_len = (((payload_len + 3) & (~0x03)) + 63 + HMAC_TAG_SIZE) & (~0x03f);

	/* Batch in the buf */
	buf->host_length[pkt_idx] = payload_len;
	if (pkt_idx == 0) {
		buf->host_pkt_offset[pkt_idx] = 0;
	}
	buf->host_pkt_offset[pkt_idx + 1] = buf->host_pkt_offset[pkt_idx] + pad_len;
	if (pkt_idx == MAX_BATCH_SIZE) {
		/* FIXME: bug when max batch size is reached */
		printf("%d, %u, %u\n", pkt_idx, buf->host_pkt_offset[pkt_idx + 1], buf->host_pkt_offset[pkt_idx]);
	}

	/*
	int i;
	for (i = payload_len; i < pad_len; i ++) {
		*(buf->host_in + buf->host_pkt_offset[pkt_idx] + i) = 0;
	}*/

	rte_memcpy(buf->host_in + buf->host_pkt_offset[pkt_idx], pkt_data, payload_len);
}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	/* TODO: update packet length with hmac tag */
	struct udp_hdr *udp = onvm_pkt_udp_hdr(pkt);
	uint8_t *pkt_data = (uint8_t *)udp + sizeof(struct udp_hdr);
	uint16_t payload_len = rte_be_to_cpu_16(udp->dgram_len) - 8;

	rte_memcpy(pkt_data, buf->host_out + buf->host_pkt_offset[pkt_idx], ((payload_len + 3) & (~0x03)) + HMAC_TAG_SIZE);
}

static void user_gpu_htod(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->dev_in, buf->host_in, buf->host_pkt_offset[job_num], ASYNC, thread_id);
	gcudaMemcpyHtoD(buf->dev_aes_key, buf->host_aes_key, job_num * AES_KEY_SIZE, ASYNC, thread_id);
	gcudaMemcpyHtoD(buf->dev_hmac_key, buf->host_hmac_key, job_num * HMAC_KEY_SIZE, ASYNC, thread_id);
	gcudaMemcpyHtoD(buf->dev_pkt_offset, buf->host_pkt_offset, (job_num + 1) * sizeof(uint32_t), ASYNC, thread_id);
	gcudaMemcpyHtoD(buf->dev_length, buf->host_length, job_num * PKT_LENGTH_SIZE, ASYNC, thread_id);
}

static void user_gpu_dtoh(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_out, buf->dev_out, buf->host_pkt_offset[job_num], ASYNC, thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	uint64_t arg_num = 8;
	uint64_t offset = 0;

	info[0] = arg_num;

	info[1] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_in), sizeof(buf->dev_in));
	offset += sizeof(buf->dev_in);

	info[2] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_out), sizeof(buf->dev_out));
	offset += sizeof(buf->dev_out);

	info[3] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_pkt_offset), sizeof(buf->dev_pkt_offset));
	offset += sizeof(buf->dev_pkt_offset);
	
	info[4] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_length), sizeof(buf->dev_length));
	offset += sizeof(buf->dev_length);
	
	info[5] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_aes_key), sizeof(buf->dev_aes_key));
	offset += sizeof(buf->dev_aes_key);
	
	info[6] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_hmac_key), sizeof(buf->dev_hmac_key));
	offset += sizeof(buf->dev_hmac_key);
	
	info[7] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(job_num), sizeof(job_num));
	offset += sizeof(job_num);

	info[8] = offset;
	*((uint8_t *)arg_buf + offset) = 0;
	offset += sizeof(void *);
}

static void init_main(void)
{
	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t) // input buffer
			+ (MAX_BATCH_SIZE + 1) * sizeof(uint32_t)  // input packet offset
			+ MAX_BATCH_SIZE * PKT_LENGTH_SIZE            // pkt length key
			+ MAX_BATCH_SIZE * AES_KEY_SIZE            // aes key
			+ MAX_BATCH_SIZE * HMAC_KEY_SIZE             // aes iv
			+ MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t), // output
			0,
			0);                                  // first time
}

static void init_gpu_schedule(void)
{
	/* Initialize the GPU info, onvm_framework_init should be performed before onvm_nflib_init */
	const char *module_file = "../ipsec/gpu/crypto.ptx";
	const char *kernel_name = "aes_ctr_sha1_kernel";
	onvm_framework_init(module_file, kernel_name);

	unsigned int pkt_size[6] = {64, 128, 256, 512, 1024, 1518};
	unsigned int line_start_batch[6] = {0, 0, 0, 0, 0, 0};
	double k1[6] = {0.024414, 0.060268, 0.136289, 0.282555, 0.570586, 1.010234};
	double b1[6] = {35.45,    47.511,   64.646,   102,      176.86,   228.12};
	double k2[6] = {0.006343, 0.023149, 0.044875, 0.085195, 0.172031, 0.249227};
	double b2[6] = {21.321,   19.404,   16.614,   16.229,   11.496,   15.257};

	onvm_framework_install_kernel_perf_para_set(k1, b1, k2, b2, pkt_size, line_start_batch, 6);
}

int main(int argc, char *argv[])
{
	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_IPSEC, &(init_gpu_schedule))) < 0)
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
