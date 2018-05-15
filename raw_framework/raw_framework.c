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

#include "onvm_nflib.h"
#include "onvm_framework.h"

#define NF_TAG "raw_framework"

typedef struct my_buf_s {
	/* Stores real data */
	uint64_t job_num;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));
	return buf;
}

static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt)
{
	UNUSED(cur_buf);
	UNUSED(pkt);
}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	UNUSED(cur_buf);
	UNUSED(pkt);
	UNUSED(pkt_idx);
}

static void user_gpu_htod(void *cur_buf, unsigned int thread_id)
{
	UNUSED(cur_buf);
	UNUSED(thread_id);
}

static void user_gpu_dtoh(void *cur_buf, unsigned int thread_id)
{
	UNUSED(cur_buf);
	UNUSED(thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info)
{
	UNUSED(cur_buf);
	UNUSED(arg_buf);
	UNUSED(arg_info);
}

static void init_main(void)
{
	;
}

static void init_gpu_schedule(void)
{
	onvm_framework_init(NULL, NULL);
}

int main(int argc, char *argv[])
{
	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_FRAMEWORK, &(init_gpu_schedule))) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	init_main();

	/* Initialization is done, start threads */
	onvm_framework_start_cpu(&(init_host_buf), &(user_batch_func), &(user_post_func));

	//onvm_framework_start_gpu(&(user_gpu_func));
	onvm_framework_start_gpu(&(user_gpu_htod), &(user_gpu_dtoh), &(user_gpu_set_arg));

	printf("If we reach here, program is ending\n");
	return 0;
}
