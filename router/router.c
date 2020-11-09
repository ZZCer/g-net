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
#include "onvm_common.h"
#include "gpu_packet.h"


#define NF_TAG "ipv4fwd"

//记录了每个数据的偏移量
uint16_t sync_offset[SYNC_DATA_COUNT]={4,4,2,2,1};

static uint8_t CR=0;
static uint8_t CW=0;
static uint8_t GR=0b11000000;
static uint8_t GW=0b11000000;

static uint16_t *tbl24_h;
static CUdeviceptr tbl24_d;

//h2d hint和d2h hint,由于不是指针，因此可以直接通过地址将数据传递给gpu
static uint8_t h2d_hint;
static uint8_t d2h_hint; 

//涉及修改headers
typedef struct my_buf_s {
	/* Stores real data */
	CUdeviceptr *host_in;
	uint8_t *host_out;

	//sourceIP desIP sourcePort desPort ProtoID 一共13字节
	char* gpu_sync;
	char* cpu_sync;

	CUdeviceptr device_in;
	CUdeviceptr device_out;
	CUdeviceptr device_sync_in;
	CUdeviceptr device_sync_out;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));

	gcudaHostAlloc((void **)&(buf->host_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaHostAlloc((void **)&(buf->host_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	gcudaHostAlloc((void **)&(buf->gpu_sync), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(char));
	gcudaHostAlloc((void **)&(buf->cpu_sync), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(char));

	gcudaMalloc(&(buf->device_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaMalloc(&(buf->device_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	gcudaMalloc(&(buf->device_sync_in), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(char));
	gcudaMalloc(&(buf->device_sync_out), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(char));

	return buf;
}

//在预处理阶段，解析出要传递的字节数组信息
static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	buf->host_in[pkt_idx] = onvm_pkt_gpu_ptr(pkt);
	uint8_t tp_h2d_hint=h2d_hint;

	int index = 0;
	while(tp_h2d_hint!=0)
	{
		index ++;
		if((tp_h2d_hint & 1) == 1)
		{
			int data_idx=SYNC_DATA_COUNT-index;
			
			switch(data_idx)
			{
				case SYNC_SOURCE_IP:
				{	
					struct ipv4_hdr* hdr1=onvm_pkt_ipv4_hdr(pkt);
					*((uint32_t*)(buf->gpu_sync))=hdr1->src_addr;
					break;
				}
				case SYNC_DEST_IP:
				{	
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct ipv4_hdr* hdr2=onvm_pkt_ipv4_hdr(pkt);
					*((uint32_t*)(buf->gpu_sync + offset))=hdr2->dst_addr;
					break;
				}
				case SYNC_SOURCE_PORT:
				{	
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct tcp_hdr* hdr3=onvm_pkt_tcp_hdr(pkt);
					*((uint16_t*)(buf->gpu_sync + offset))=hdr3->src_port;
					break;
				}
				case SYNC_DEST_PORT:
				{
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct tcp_hdr* hdr4=onvm_pkt_tcp_hdr(pkt);
					*((uint16_t*)(buf->gpu_sync + offset))=hdr4->dst_port;
					break;
				}
				case SYNC_TCP_FLAGS:
				{
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct tcp_hdr* hdr5=onvm_pkt_tcp_hdr(pkt);
					*((uint8_t*)(buf->gpu_sync + offset))=hdr5->tcp_flags;
					break;
				}
				default:
					break;
			}
		}
		tp_h2d_hint>>=1;
	}
}

//再
static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	/* Write the port */
	pkt->port = buf->host_out[pkt_idx];

	//这里面套个switch就结束了
	//在d2h后，数据就已经从device获取到了
	uint8_t tp_d2h_hint = buf->host_out;
	int index = 0;

	//根据d2h_hint，进行遍历，将其中每个位的数据解析出来
	while (tp_d2h_hint!=0)
	{
		index ++;
		if(tp_d2h_hint & 1 == 1)
		{
			int data_idx=SYNC_DATA_COUNT-index;
			switch(data_idx)
			{
				case SYNC_SOURCE_IP:
				{	
					struct ipv4_hdr* hdr1=onvm_pkt_ipv4_hdr(pkt);
					hdr1->src_addr = *((uint32_t*)(buf->gpu_sync));
					break;
				}
				case SYNC_DEST_IP:
				{
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct ipv4_hdr* hdr2=onvm_pkt_ipv4_hdr(pkt);
					hdr2->dst_addr = *((uint32_t*)(buf->gpu_sync + offset));
					break;
				}
				case SYNC_SOURCE_PORT:
				{
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct tcp_hdr* hdr3=onvm_pkt_tcp_hdr(pkt);
					hdr3->src_port = *((uint16_t*)(buf->gpu_sync + offset));
					break;
				}
				case SYNC_DEST_PORT:
				{
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct tcp_hdr* hdr4=onvm_pkt_tcp_hdr(pkt);
					hdr4->dst_port = *((uint16_t*)(buf->gpu_sync + offset));
					break;
				}
				case SYNC_TCP_FLAGS:
				{
					int offset = 0;
					for(int i=0;i<data_idx;i++)
						offset += sync_offset[i];

					struct tcp_hdr* hdr5=onvm_pkt_tcp_hdr(pkt);
					hdr5->tcp_flags = *((uint8_t*)(buf->gpu_sync + offset));
					break;
				}
				default:
					break;
			}
		}
		tp_d2h_hint>>=1;
	}	
}

static void user_gpu_htod(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->device_in, buf->host_in, job_num * sizeof(CUdeviceptr), ASYNC, thread_id);
	gcudaMemcpyHtoD(buf->device_sync_in, buf->gpu_sync, job_num * SYNC_DATA_SIZE, ASYNC, thread_id);
}

static void user_gpu_dtoh(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_out, buf->device_out, job_num * sizeof(uint8_t), ASYNC, thread_id);
	gcudaMemcpyDtoH(buf->cpu_sync, buf->device_sync_out, job_num * SYNC_DATA_SIZE, ASYNC, thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	//tmd 这个参数忘记改 花了我一晚上时间debug
	uint64_t arg_num = 8;
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
	
	info[4] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(tbl24_d), sizeof(tbl24_d));
	offset += sizeof(tbl24_d);

	//同步相关参数
	info[5] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &h2d_hint, sizeof(h2d_hint));
	offset += sizeof(h2d_hint);

	info[6] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &d2h_hint, sizeof(d2h_hint));
	offset += sizeof(d2h_hint);

	info[7] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->device_sync_in) , sizeof(buf->device_sync_in)) ;
	offset += sizeof(buf->device_sync_in);

	info[8] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->device_sync_out) , sizeof(buf->device_sync_out)) ;
	offset += sizeof(buf->device_sync_out);
}

static void init_main(void)
{
	int table_item_num = 1 << 24;

	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * sizeof(CUdeviceptr)  // host_in
			+ MAX_BATCH_SIZE * sizeof(uint8_t) 			 // host_out
			+ MAX_BATCH_SIZE*SYNC_DATA_SIZE*sizeof(char) // h2d sync data
			+ MAX_BATCH_SIZE*SYNC_DATA_SIZE*sizeof(char) // d2h sync data		 	
			,
			table_item_num * sizeof(uint16_t)
			+ MAX_BATCH_SIZE * sizeof(uint8_t)*2,       //d2h hints & h2d hints 
			0);                                       // first time

	gcudaMalloc(&tbl24_d, table_item_num * sizeof(uint16_t));
	gcudaHostAlloc((void **)&tbl24_h, table_item_num * sizeof(uint16_t));

	int i;
	for (i = 0; i < table_item_num; i ++) {
		tbl24_h[i] = i & (0xffff);
	}

	gcudaMemcpyHtoD(tbl24_d, tbl24_h, table_item_num * sizeof(uint16_t), SYNC, 0);
}

static void init_gpu_schedule(void)
{
	/* Initialize the GPU info, onvm_framework_init should be performed before onvm_nflib_init */
	const char *module_file = "../router/gpu/ipv4lookup.ptx";
	const char *kernel_name = "ipv4lookup";
	onvm_framework_init(module_file, kernel_name);

	double K1 = 0.00109625;
	double B1 = 8.425;
	double K2 = 0.0004133;
	double B2 = 2.8036;

	onvm_framework_install_kernel_perf_parameter(K1, B1, K2, B2);
}

int main(int argc, char *argv[])
{
	hints hint;
	hint.CR=CR;
	hint.CW=CW;
	hint.GR=GR;
	hint.GW=GW;

	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, hint ,NF_TAG, NF_ROUTER,GPU_NF,&(init_gpu_schedule))) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	init_main();

	//获取hint信息
	onvm_framework_get_hint(&h2d_hint,&d2h_hint);

	printf("EAL: h2d_hint:%d d2h_hint:%d\n",h2d_hint,d2h_hint);

	/* Initialization is done, start threads */
	onvm_framework_start_cpu(&(init_host_buf), &(user_batch_func), &(user_post_func),NULL,GPU_NF);

	onvm_framework_start_gpu(&(user_gpu_htod), &(user_gpu_dtoh), &(user_gpu_set_arg));

	printf("If we reach here, program is ending\n");
	return 0;
}
