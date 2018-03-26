#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#include "libgpuipv4lookup.h"

#define NUM_LOOP 10

//#define KERNEL_TEST 1
#define NO_STREAM 1

int main(int argc, char*argv[])
{
	//assert(cudaSetDevice(1) == cudaSuccess);
	int i, j, stream_id;
	struct timespec start, end;
#if defined(KERNEL_TEST)
	struct timespec kernel_start, kernel_end;
#endif

	unsigned int NUM_PKTS = 1024, NUM_BLKS = 1, THREADS_PER_BLK = 1024, STREAM_NUM = 1;
	if (argc == 2) {
		NUM_PKTS = atoi(argv[1]);
	} else if (argc == 3) {
		NUM_PKTS = atoi(argv[1]);
		NUM_BLKS = atoi(argv[2]);
	} else if (argc == 4) {
		NUM_PKTS = atoi(argv[1]);
		NUM_BLKS = atoi(argv[2]);
		STREAM_NUM = atoi(argv[3]);
	}

	THREADS_PER_BLK = NUM_PKTS / NUM_BLKS;
	if (THREADS_PER_BLK > 1024) THREADS_PER_BLK = 1024;

	//printf("Num of flows is %d, threads_per_blk is %d, num of blocks is %d, stream num is %d\n", 
	//		NUM_PKTS, THREADS_PER_BLK, NUM_BLKS, STREAM_NUM);
	//printf("###############################################\n");

#if !defined(NO_STREAM)
	cudaStream_t stream[STREAM_NUM];
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaStreamCreate(&stream[i]);
	}
#endif

	uint32_t *host_in[STREAM_NUM], *device_in[STREAM_NUM];
	uint8_t *host_out[STREAM_NUM], *device_out[STREAM_NUM];
	uint16_t *tbl24_h, *tbl24_d;
	double diff, total_diff = 0;

	assert (STREAM_NUM == 1);
	stream_id = 0;

	for (i = 0; i < STREAM_NUM; i ++) {
		cudaHostAlloc((void **)&(host_in[i]), NUM_PKTS * sizeof(uint32_t), cudaHostAllocDefault);
		for (j = 0; j < NUM_PKTS; j ++) {
			host_in[i][j] = rand() << 8;
		}
		cudaHostAlloc((void **)&(host_out[i]), NUM_PKTS * sizeof(uint8_t), cudaHostAllocDefault);

		cudaMalloc((void **)&(device_in[i]), NUM_PKTS * sizeof(uint32_t));
		cudaMalloc((void **)&(device_out[i]), NUM_PKTS * sizeof(uint8_t));
	}

	cudaHostAlloc((void **)&tbl24_h, (1 << 24) * sizeof(uint16_t), cudaHostAllocDefault);
	cudaMalloc((void **)&tbl24_d, (1 << 24) * sizeof(uint16_t));
	for (i = 0; i < (1 << 24); i ++) {
		tbl24_h[i] = i & (0xffff);
	}
	

	cudaMemcpy(tbl24_d, tbl24_h, (1 << 24) * sizeof(uint16_t), cudaMemcpyHostToDevice);

	/* warm up */
#if defined(NO_STREAM)
		cudaMemcpyAsync(device_in[stream_id], host_in[stream_id], NUM_PKTS * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);

		IPv4_Lookup(device_in[stream_id],
			    NUM_PKTS,
			    device_out[stream_id],
			    tbl24_d,
			    THREADS_PER_BLK,
			    NUM_BLKS,
			    0);

		cudaDeviceSynchronize();
#else
	for (stream_id = 0; stream_id < STREAM_NUM; stream_id ++) {
		cudaMemcpyAsync(device_in[stream_id], host_in[stream_id], NUM_PKTS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream[stream_id]);

		IPv4_Lookup(device_in[stream_id],
			    NUM_PKTS,
			    device_out[stream_id],
			    tbl24_d,
			    THREADS_PER_BLK,
			    NUM_BLKS,
			    stream[stream_id]);

		cudaDeviceSynchronize();
	}
#endif

	/* Real test */
	for (i = 0; i < NUM_LOOP; i ++) {
		clock_gettime(CLOCK_MONOTONIC, &start);


#if defined(NO_STREAM)
			cudaMemcpyAsync(device_in[stream_id], host_in[stream_id], NUM_PKTS * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
#else
		for (stream_id = 0; stream_id < STREAM_NUM; stream_id ++) {
			cudaMemcpyAsync(device_in[stream_id], host_in[stream_id], NUM_PKTS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream[stream_id]);
#endif

#if defined(KERNEL_TEST)
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif

#if defined(NO_STREAM)
			IPv4_Lookup(device_in[stream_id],
					NUM_PKTS,
					device_out[stream_id],
					tbl24_d,
					THREADS_PER_BLK,
					NUM_BLKS,
					0);
#else
			IPv4_Lookup(device_in[stream_id],
					NUM_PKTS,
					device_out[stream_id],
					tbl24_d,
					THREADS_PER_BLK,
					NUM_BLKS,
					stream[stream_id]);
#endif

#if defined(KERNEL_TEST)
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_MONOTONIC, &kernel_end);
#endif
			
#if defined(NO_STREAM)
			cudaMemcpyAsync(host_out[stream_id], device_out[stream_id], NUM_PKTS * sizeof(uint8_t), cudaMemcpyDeviceToHost, 0);
#else
			cudaMemcpyAsync(host_out[stream_id], device_out[stream_id], NUM_PKTS * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[stream_id]);
		}
#endif

		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end);

#if defined(KERNEL_TEST)
		diff = 1000000 * (kernel_end.tv_sec-kernel_start.tv_sec) + (kernel_end.tv_nsec-kernel_start.tv_nsec)/1000;
		total_diff += diff;
#else
		diff = 1000000 * (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/1000;
		total_diff += diff;
#endif
	}

/*
	uint32_t hash;
	uint16_t value_tb1;
	for (i = 0; i < NUM_PKTS; i++){
		printf("in %x, out %x\t", host_in[0][i], host_out[0][i]);
		hash = host_in[0][i] >> 8;
		value_tb1 = tbl24_h[hash];
		printf("[%x] %x %x\n", hash, value_tb1, (uint8_t)value_tb1);
	}
*/

	//printf("[pkt num: %4d]: \t %.2lf us, \t %.2lf Mpps\n", NUM_PKTS, (double)total_diff/NUM_LOOP, (double)(NUM_PKTS * STREAM_NUM * NUM_LOOP) / total_diff);
#if defined(NO_STREAM)
	printf("%4d\t%.2lf\t%.2lf\n", NUM_PKTS, (double)total_diff/NUM_LOOP, (double)(NUM_PKTS * NUM_LOOP) / total_diff);
#else
	printf("%4d\t%.2lf\t%.2lf\n", NUM_PKTS, (double)total_diff/NUM_LOOP, (double)(NUM_PKTS * STREAM_NUM * NUM_LOOP) / total_diff);
#endif

	return 0;
}
