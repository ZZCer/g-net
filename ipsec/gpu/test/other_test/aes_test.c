#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#include "crypto_size.h"
#include "libgpucrypto.h"

#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#define NUM_LOOP 10

//#define KERNEL_TEST 1

int main(int argc, char*argv[])
{
	FILE *fp;
	uint16_t i, fsize, pad_size, stream_id;
	char * rtp_pkt;
	uint8_t default_aes_keys[AES_KEY_SIZE], default_ivs[AES_IV_SIZE];

	struct  timespec start, end;
#if defined(KERNEL_TEST)
	struct  timespec kernel_start, kernel_end;
#endif


	unsigned int NUM_FLOWS = 1024, NUM_BLKS = 1, THREADS_PER_BLK = 1024, STREAM_NUM = 1;
	unsigned int PKT_SIZE = 64;
	if (argc == 3) {
		NUM_FLOWS = atoi(argv[1]);
		PKT_SIZE = atoi(argv[2]);
	} else if (argc == 4) {
		NUM_FLOWS = atoi(argv[1]);
		PKT_SIZE = atoi(argv[2]);
		NUM_BLKS = atoi(argv[3]); /* thread blocks per stream */
	} else if (argc == 5) {
		NUM_FLOWS = atoi(argv[1]); /* job_num per stream */
		PKT_SIZE = atoi(argv[2]);
		STREAM_NUM = atoi(argv[3]);
		NUM_BLKS = atoi(argv[4]); /* thread blocks per stream */
	}
	THREADS_PER_BLK = NUM_FLOWS / NUM_BLKS;
	if (NUM_FLOWS > 1024)	THREADS_PER_BLK = 1024;
	//printf("Num of flows is %d, threads_per_blk is %d, num of blocks is %d, pkt size is %d, stream num is %d\n", 
	//	NUM_FLOWS, THREADS_PER_BLK, NUM_BLKS, PKT_SIZE, STREAM_NUM);
	//printf("###############################################\n");

	cudaStream_t stream[STREAM_NUM];
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaStreamCreate(&stream[i]);
	}

	uint8_t *host_in, *host_out[STREAM_NUM], *device_in[STREAM_NUM], *device_out[STREAM_NUM];
	uint8_t *host_aes_keys, *device_aes_keys[STREAM_NUM];
	uint8_t *host_ivs, *device_ivs[STREAM_NUM];
	uint32_t *host_pkt_offset, *device_pkt_offset[STREAM_NUM];

	double diff, total_diff = 0;
	uint8_t a = 123; //random

	fp = fopen("rtp.pkt", "rb");
	fseek(fp, 0, SEEK_END);
	// NOTE: fsize should be 1356 bytes
	//fsize = ftell(fp);
	fsize = 1328;
	fseek(fp, 0, SEEK_SET);

	rtp_pkt = (char *)calloc(fsize, sizeof(char));
	int n = fread(rtp_pkt, fsize, sizeof(char), fp);
	if (n != 1) printf("read error\n");

	fsize = PKT_SIZE;
	pad_size = (PKT_SIZE + 15) & (~0x0f);

	for (i = 0; i < AES_KEY_SIZE; i ++)
		default_aes_keys[i] = a;
	for (i = 0; i < AES_IV_SIZE; i ++)
		default_ivs[i] = a;

	//printf("duplicate it %d times, takes %d bytes\n",NUM_FLOWS,pad_size*NUM_FLOWS);
	cudaHostAlloc((void **)&host_in, pad_size * NUM_FLOWS * sizeof(uint8_t), cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_aes_keys, NUM_FLOWS * AES_KEY_SIZE, cudaHostAllocWriteCombined);
	cudaHostAlloc((void **)&host_ivs, NUM_FLOWS * AES_IV_SIZE, cudaHostAllocWriteCombined);
	cudaHostAlloc((void **)&host_pkt_offset, (NUM_FLOWS + 1) * PKT_OFFSET_SIZE, cudaHostAllocWriteCombined);

	for (i = 0; i < NUM_FLOWS; i ++){
		memcpy(host_in + i * pad_size, rtp_pkt, fsize * sizeof(uint8_t));
		memcpy((uint8_t *)host_aes_keys + i * AES_KEY_SIZE, default_aes_keys, AES_KEY_SIZE);
		memcpy((uint8_t *)host_ivs + i * AES_IV_SIZE, default_ivs, AES_IV_SIZE);
		host_pkt_offset[i] = i * pad_size;
	}
	host_pkt_offset[NUM_FLOWS] = NUM_FLOWS * pad_size;


	for (i = 0; i < STREAM_NUM; i ++) {
		cudaHostAlloc((void **)&host_out[i], pad_size * NUM_FLOWS * sizeof(uint8_t), cudaHostAllocDefault);

		cudaMalloc((void **)&(device_in[i]), pad_size * NUM_FLOWS * sizeof(uint8_t));
		cudaMalloc((void **)&(device_out[i]), pad_size * NUM_FLOWS * sizeof(uint8_t));
		cudaMalloc((void **)&(device_aes_keys[i]), NUM_FLOWS * AES_KEY_SIZE);
		cudaMalloc((void **)&(device_ivs[i]), NUM_FLOWS * AES_IV_SIZE);
		cudaMalloc((void **)&(device_pkt_offset[i]), (NUM_FLOWS + 1) * PKT_OFFSET_SIZE);
	}

	/* warm up */
	for (stream_id = 0; stream_id < STREAM_NUM; stream_id ++) {
		cudaMemcpyAsync(device_in[stream_id], host_in, pad_size * NUM_FLOWS * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[stream_id]);
		cudaMemcpyAsync(device_aes_keys[stream_id], host_aes_keys, NUM_FLOWS * AES_KEY_SIZE, cudaMemcpyHostToDevice, stream[stream_id]);
		cudaMemcpyAsync(device_ivs[stream_id], host_ivs, NUM_FLOWS * AES_IV_SIZE, cudaMemcpyHostToDevice, stream[stream_id]);
		cudaMemcpyAsync(device_pkt_offset[stream_id], host_pkt_offset, (NUM_FLOWS + 1) * PKT_OFFSET_SIZE, cudaMemcpyHostToDevice, stream[stream_id]);

		AES_cbc_128_encrypt_gpu(
				device_in[stream_id],
				device_out[stream_id],
				device_pkt_offset[stream_id],
				device_aes_keys[stream_id],
				device_ivs[stream_id],
				NUM_FLOWS,
				NULL,
				THREADS_PER_BLK,
				NUM_BLKS,
				stream[stream_id]);

		cudaDeviceSynchronize();
	}

	/* Real test */
	for (i = 0; i < NUM_LOOP; i ++) {
		clock_gettime(CLOCK_MONOTONIC, &start);

		for (stream_id = 0; stream_id < STREAM_NUM; stream_id ++) {

			cudaMemcpyAsync(device_in[stream_id], host_in, pad_size * NUM_FLOWS * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[stream_id]);
			cudaMemcpyAsync(device_aes_keys[stream_id], host_aes_keys, NUM_FLOWS * AES_KEY_SIZE, cudaMemcpyHostToDevice, stream[stream_id]);
			cudaMemcpyAsync(device_ivs[stream_id], host_ivs, NUM_FLOWS * AES_IV_SIZE, cudaMemcpyHostToDevice, stream[stream_id]);
			cudaMemcpyAsync(device_pkt_offset[stream_id], host_pkt_offset, (NUM_FLOWS + 1) * PKT_OFFSET_SIZE, cudaMemcpyHostToDevice, stream[stream_id]);

#if defined(KERNEL_TEST)
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif
			AES_cbc_128_encrypt_gpu(
					device_in[stream_id],
					device_out[stream_id],
					device_pkt_offset[stream_id],
					device_aes_keys[stream_id],
					device_ivs[stream_id],
					NUM_FLOWS,
					NULL,
					THREADS_PER_BLK,
					NUM_BLKS,
					stream[stream_id]);
#if defined(KERNEL_TEST)
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_MONOTONIC, &kernel_end);
#endif
			cudaMemcpyAsync(host_out[stream_id], device_out[stream_id], pad_size * NUM_FLOWS * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[stream_id]);
		}

		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end);

#if defined(KERNEL_TEST)
		diff = 1000000 * (kernel_end.tv_sec-kernel_start.tv_sec)+ (kernel_end.tv_nsec-kernel_start.tv_nsec)/1000;
		total_diff += diff;
		// printf("Only Kernel, the difference is %lf ms, speed is %lf Mbps\n", (double)diff/1000, (double)((fsize * 8) * NUM_FLOWS * STREAM_NUM) / diff);
#else
		diff = 1000000 * (end.tv_sec-start.tv_sec)+ (end.tv_nsec-start.tv_nsec)/1000;
		total_diff += diff;
		//printf("%.2lf us\n", (double)diff);
		//printf("%.2lf us, %.2lf Mbps\n", (double)diff, (double)((fsize * 8) * NUM_FLOWS * STREAM_NUM) / diff);
#endif
	}

//	printf("[pkt_size: %4d, pkt num: %4d]: \t %.2lf us, \t %.2lf Gbps\n", PKT_SIZE, NUM_FLOWS, (double)total_diff/NUM_LOOP, (double)(NUM_LOOP * (fsize * 8) * NUM_FLOWS * STREAM_NUM) / (total_diff * 1000.0));
	/* Calculate Gbps */
	//printf("%4d\t%.2lf\t%.2lf\n", NUM_FLOWS, (double)total_diff/NUM_LOOP, (double)(NUM_LOOP * (fsize * 8) * NUM_FLOWS * STREAM_NUM) / (total_diff * 1000.0));
	/* Calculate Mpps */
	printf("%4d\t%.2lf\t%.2lf\n", NUM_FLOWS, (double)total_diff/NUM_LOOP, (double)(NUM_LOOP * NUM_FLOWS * STREAM_NUM) / total_diff);
	//printf("%.2lf\n", (double)total_diff/NUM_LOOP);

	return 0;
}
