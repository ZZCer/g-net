#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "firewall_kernel.h"
#include "libgpufirewall.h"

#define NUM_LOOP 10
//#define KERNEL_TEST 0

static void construct_rules(struct fwRule *rules)
{
	FILE * f = freopen("data.in", "r", stdin);
	if (f == NULL) {
		printf("file open error\n");
	}

	int i;
	struct inputItem *in = (struct inputItem *)malloc(RULESIZE * sizeof(struct inputItem));

	for (i = 0; i < RULESIZE; i ++) {
		in[i].srcAddr[0] = 0;
		in[i].srcAddr[1] = 0;
		in[i].srcAddr[2] = 0;
		in[i].srcAddr[3] = 0;
		in[i].srcMask = 0;
		in[i].desAddr[0] = 0;
		in[i].desAddr[1] = 0;
		in[i].desAddr[2] = 0;
		in[i].desAddr[3] = 0;
		in[i].desMask = 0;

		in[i].srcPort[0] = 0;
		in[i].srcPort[1] = 0;
		in[i].desPort[0] = 0;
		in[i].desPort[1] = 0;

		in[i].aChar[0] = '\0';
		in[i].aChar[1] = '\0';
		in[i].aChar[2] = '\0';
		in[i].aChar[3] = '\0';
		in[i].bChar[0] = '\0';
		in[i].bChar[1] = '\0';
		in[i].bChar[2] = '\0';
		in[i].bChar[3] = '\0';
		in[i].bChar[4] = '\0';
		in[i].bChar[5] = '\0';
		in[i].bChar[6] = '\0';
		in[i].bChar[7] = '\0';
	}

	for (i = 0; i < RULESIZE; i ++) {
		int f = scanf("@%d.%d.%d.%d/%d\t%d.%d.%d.%d/%d\t%d : %d\t%d : %d\t0x%c%c/0x%c%c\t0x%c%c%c%c/0x%c%c%c%c\n", 
				&in[i].srcAddr[0], &in[i].srcAddr[1], &in[i].srcAddr[2], &in[i].srcAddr[3], &in[i].srcMask,
				&in[i].desAddr[0], &in[i].desAddr[1], &in[i].desAddr[2], &in[i].desAddr[3], &in[i].desMask,
				&in[i].srcPort[0], &in[i].srcPort[1], &in[i].desPort[0], &in[i].desPort[1], 
				&in[i].aChar[0], &in[i].aChar[1], &in[i].aChar[2], &in[i].aChar[3], 
				&in[i].bChar[0], &in[i].bChar[1], &in[i].bChar[2], &in[i].bChar[3], 
				&in[i].bChar[4], &in[i].bChar[5], &in[i].bChar[6], &in[i].bChar[7]);
		if (f == 0) {
			printf("scanf error\n");
		}
	}

	for (i = 0; i < RULESIZE; i ++)
	{
		rules[i].rule.srcAddr = (in[i].srcAddr[0] << 24) | (in[i].srcAddr[1] << 16) | (in[i].srcAddr[2] << 8) | in[i].srcAddr[3];
		rules[i].rule.desAddr = (in[i].desAddr[0] << 24) | (in[i].desAddr[1] << 16) | (in[i].desAddr[2] << 8) | in[i].desAddr[3];
		rules[i].rule.srcMask = in[i].srcMask;
		rules[i].rule.desMask = in[i].desMask;
		rules[i].rule.srcPortStart = in[i].srcPort[0];
		rules[i].rule.srcPortEnd = in[i].srcPort[1];
		rules[i].rule.desPortStart = in[i].desPort[0];
		rules[i].rule.desPortEnd = in[i].desPort[1];

		rules[i].rule.protocol = rand() % 4;
	}

	firewall_rule_construct(rules, RULESIZE, 0);
}

int main(int argc, char *argv[])
{
	int block_num = 1, threads_per_blk = 1024, batch_size = 1024;
	double diff, total_diff = 0;

	if (argc == 3) {
		batch_size = atoi(argv[1]);
		block_num = atoi(argv[2]);
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

	struct pcktFive *pcktFwFives = NULL, *dev_pcktFwFives = NULL;
	unsigned int *res = NULL, *dev_res = NULL;

	struct fwRule *rules = NULL;
	int i;

	struct  timespec start, end;
#if defined(KERNEL_TEST)
	struct  timespec kernel_start, kernel_end;
#endif

	/* construct random packets */
	pcktFwFives = (struct pcktFive *)malloc(batch_size * sizeof(struct pcktFive));
	for (i = 0; i < batch_size; i ++) {
		pcktFwFives[i].srcAddr = rand();
		pcktFwFives[i].desAddr = rand();
		pcktFwFives[i].srcPort = rand();
		pcktFwFives[i].desPort = rand();
		pcktFwFives[i].protocol = TYPE_TCP;

	}

	/* read and build rules */
	rules = (struct fwRule *)malloc(RULESIZE * sizeof(struct fwRule));
	for (i = 0; i < RULESIZE; i ++)
	{
		rules[i].rule.srcAddr = 0;
		rules[i].rule.desAddr = 0;
		rules[i].rule.srcMask = 0;
		rules[i].rule.desMask = 0;
		rules[i].rule.srcPortStart = 0;
		rules[i].rule.srcPortEnd = 0;
		rules[i].rule.desPortStart = 0;
		rules[i].rule.desPortEnd = 0;
		rules[i].rule.protocol = 0;
		rules[i].order = i;
		rules[i].action = rand() % 2;

	}
	construct_rules(rules);

	/* allocate and clear results */
	res = (unsigned int *)malloc(batch_size * 4 * sizeof(unsigned int));
	for (i = 0; i < batch_size * 4; i ++) {
		res[i] = 0;
	}

	/* cuda malloc */
	cudaMalloc((void**)&dev_pcktFwFives, batch_size * sizeof(struct pcktFive));
	cudaMalloc((void**)&dev_res, batch_size * 4 * sizeof(unsigned int));


	/* warm up */
	cudaMemcpy(dev_pcktFwFives, pcktFwFives, batch_size * sizeof(struct pcktFive), cudaMemcpyHostToDevice);
	firewall_kernel(dev_pcktFwFives, dev_res, batch_size, block_num, threads_per_blk);
	cudaMemcpy(res, dev_res, batch_size * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (i = 0; i < NUM_LOOP; i ++) {
		clock_gettime(CLOCK_MONOTONIC, &start);

		cudaMemcpy(dev_pcktFwFives, pcktFwFives, batch_size * sizeof(struct pcktFive), cudaMemcpyHostToDevice);
#if defined(KERNEL_TEST)
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif
		/* launch kernel */
		firewall_kernel(dev_pcktFwFives, dev_res, batch_size, block_num, threads_per_blk);

#if defined(KERNEL_TEST)
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &kernel_end);
#endif
		cudaMemcpy(res, dev_res, batch_size * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

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

	//printf("[pkt num: %4d]: \t %.2lf us, \t %.2lf Mpps\n", batch_size, (double)total_diff/NUM_LOOP, 
	//	(double)(NUM_LOOP * batch_size) / total_diff);
	printf("%4d\t%.2lf\t%.2lf\n", batch_size, (double)total_diff/NUM_LOOP, 
		(double)(NUM_LOOP * batch_size) / total_diff);

	return 0;
}
