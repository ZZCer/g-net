#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime.h>

#include "rules.h"
#include "libgpuids.h"

#define NUM_LOOP 10
//#define KERNEL_TEST 1

#define CUDA_SAFE_CALL(call) \
do { \
	cudaError_t err = call; \
		if (cudaSuccess != err) { \
			fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
					__FILE__, __LINE__, cudaGetErrorString(err) ); \
				exit(EXIT_FAILURE); \
		} \
} while (0)

void gen_pkts(char *packets, int batch_size, uint32_t PKT_LEN)
{
	int i, j, k;

	for (i = 0; i < batch_size; i ++) {
		k = 0;

		for (j = 0; j < PKT_LEN; j ++) {
			packets[i * PKT_LEN + j] = (char)(97 + k);
			if (k == 25) k = -1;
			k++;
		}
	}
}

int main(int argc, char *argv[])
{
	int block_num = 1, threads_per_blk = 1024, batch_size = 1024;
	//uint32_t PKT_LEN = 1518;
	uint32_t PKT_LEN = 128;
	/* the length of the packet buffer */
	uint32_t MAX_LEN = 1520;

	char *packets;
	uint16_t *host_res;
	uint32_t *pkt_offset;

	uint16_t *dev_acGPU;
	char *dev_pkt;
	uint16_t *dev_res = NULL;
	uint32_t *dev_pkt_offset = NULL;

	double diff, total_diff = 0;
	struct  timespec start, end;
#if defined(KERNEL_TEST)
	struct  timespec kernel_start, kernel_end;
#endif

	if (argc == 4) {
		batch_size = atoi(argv[1]);
		PKT_LEN = atoi(argv[2]);
		block_num = atoi(argv[3]);
	} else if (argc == 3) {
		batch_size = atoi(argv[1]);
		PKT_LEN = atoi(argv[2]);
	} else if (argc == 2) {
		batch_size = atoi(argv[1]);
	} else if (argc > 1) {
		printf("error, existing...\n");
		exit(0);
	}

	threads_per_blk = batch_size / block_num;
	if (threads_per_blk > 1024) {
		threads_per_blk = 1024;
	}

	packets = (char *)malloc(batch_size * MAX_LEN * sizeof(char));
	pkt_offset = (uint32_t *)malloc((batch_size + 1) * sizeof(uint32_t));
	int i;
	for (i = 0; i < batch_size; i ++){
		pkt_offset[i] = i * PKT_LEN;
	}
	pkt_offset[batch_size] = batch_size * PKT_LEN;
	
	host_res = (uint16_t *)malloc(batch_size * sizeof(uint16_t));
	memset(host_res, 0, batch_size * sizeof(uint16_t));

	ListRoot *listroot;
	//listroot = configrules("community.rules");
	listroot = configrules("test-rules");
	precreatearray(listroot);
	// test(listroot);

	gen_pkts(packets, batch_size, PKT_LEN);

	RuleSetRoot *rsr = listroot->TcpListRoot->prmGeneric->rsr;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_acGPU, MAX_STATE * 257 * sizeof(uint16_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_pkt, batch_size * MAX_LEN * sizeof(char)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_res, batch_size * sizeof(uint16_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&(dev_pkt_offset), (batch_size + 1) * sizeof(uint32_t)));

	CUDA_SAFE_CALL(cudaMemcpy(dev_acGPU, rsr->acGPU, MAX_STATE * 257 * sizeof(uint16_t), cudaMemcpyHostToDevice));

	/* warm up */
	CUDA_SAFE_CALL(cudaMemcpy(dev_pkt, packets, batch_size * PKT_LEN * sizeof(char), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_pkt_offset, pkt_offset, (batch_size + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
	gpumatch(block_num, threads_per_blk, batch_size, dev_acGPU, dev_pkt, dev_pkt_offset, dev_res);
	CUDA_SAFE_CALL(cudaMemcpy(host_res, dev_res, batch_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

	for (i = 0; i < NUM_LOOP; i ++) {
		clock_gettime(CLOCK_MONOTONIC, &start);

		CUDA_SAFE_CALL(cudaMemcpy(dev_pkt, packets, batch_size * PKT_LEN * sizeof(char), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(dev_pkt_offset, pkt_offset, (batch_size + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
#if defined(KERNEL_TEST)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif

		gpumatch(block_num, threads_per_blk, batch_size, dev_acGPU, dev_pkt, dev_pkt_offset, dev_res);

#if defined(KERNEL_TEST)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &kernel_end);
#endif
		CUDA_SAFE_CALL(cudaMemcpy(host_res, dev_res, batch_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC, &end);

#if defined(KERNEL_TEST)
		diff = 1000000 * (kernel_end.tv_sec-kernel_start.tv_sec)+ (kernel_end.tv_nsec-kernel_start.tv_nsec)/1000;
		total_diff += diff;
#else
		diff = 1000000 * (end.tv_sec-start.tv_sec)+ (end.tv_nsec-start.tv_nsec)/1000;
		total_diff += diff;
#endif
	}

	//printf("[pkt num: %4d]: \t %.2lf us, \t %.2lf Gbps\n", batch_size, (double)total_diff/NUM_LOOP, 
	//	(double)(NUM_LOOP * batch_size * PKT_LEN * 8) / (total_diff * 1000.0));
	printf("%4d\t%.2lf\t%.2lf\n", batch_size, (double)total_diff/NUM_LOOP, 
		(double)(NUM_LOOP * batch_size * PKT_LEN * 8) / (total_diff * 1000.0));
	//printf("%.2lf\n", (double)total_diff/NUM_LOOP); 

	freeall(listroot);
	cudaFree(dev_acGPU);
	cudaFree(dev_pkt);
	cudaFree(dev_res);
	cudaFree(dev_pkt_offset);

	return 0;
}
