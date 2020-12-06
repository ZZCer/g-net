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

//从srcIP开始，用来记录需要同步的数据的offset,同时有一个sync_num用来记录需要同步的数据个数
uint16_t *h2d_offset;
uint16_t *d2h_offset;
//包含了h2d d2h header payload等同步的信息
packet_sync_global_t pkt_sync_global;

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

	CUdeviceptr device_in;
	CUdeviceptr device_out;

	//这个pkt_sync中专门存储了用来进行h2d d2h同步的一些结构
	packet_sync_t pkt_sync;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));

	gcudaHostAlloc((void **)&(buf->host_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaHostAlloc((void **)&(buf->host_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	gcudaHostAlloc((void **)&(buf->pkt_sync.hdr_sync_h2d), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	gcudaHostAlloc((void **)&(buf->pkt_sync.hdr_sync_d2h), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	gcudaHostAlloc((void **)&(buf->pkt_sync.pld_sync_h2d), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(char));
	gcudaHostAlloc((void **)&(buf->pkt_sync.pld_sync_d2h), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(char));


	gcudaMalloc(&(buf->device_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaMalloc(&(buf->device_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	gcudaMalloc(&(buf->pkt_sync.d_hdr_sync_h2d), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	gcudaMalloc(&(buf->pkt_sync.d_hdr_sync_d2h), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	gcudaMalloc(&(buf->pkt_sync.d_pld_sync_h2d), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t));
	gcudaMalloc(&(buf->pkt_sync.d_pld_sync_d2h), MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(uint8_t));
	
	return buf;
}

//在预处理阶段，解析出要传递的字节数组信息
//取数据是没有办法优化掉的，但是获取我要取那些数据，这个是可以优化掉的。
static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	//获取payload_size，目前系统中的数据包大小都是固定数据
	char* pld_start = onvm_pkt_payload(pkt,&pkt_sync_global.payload_size);
	uint32_t pld_size = pkt_sync_global.payload_size;

	buf->host_in[pkt_idx] = onvm_pkt_gpu_ptr(pkt);

	//首部同步
	int offset = pkt_idx * SYNC_DATA_SIZE;

	struct ipv4_hdr* ip_h=onvm_pkt_ipv4_hdr(pkt);
	struct tcp_hdr* tcp_h=onvm_pkt_tcp_hdr(pkt);
	struct udp_hdr* udp_h=onvm_pkt_udp_hdr(pkt);

	uint16_t srcPort = 0;
	uint16_t dstPort = 0;

	if(tcp_h != NULL)
	{
		srcPort = tcp_h->src_port;
		dstPort = tcp_h->dst_port;
	}	
	else
	{
		srcPort = udp_h->src_port;
		dstPort = udp_h->dst_port;
	}
	
	for(size_t i=0 ; i < pkt_sync_global.h2d_sync_num ; i++)
	{
		if(h2d_offset[i] <= 4)			
			*((uint32_t*)(buf->pkt_sync.hdr_sync_h2d + offset + h2d_offset[i])) = ( (h2d_offset[i] == 4) ? ip_h->dst_addr : ip_h->src_addr );
		else if(h2d_offset[i] <= 10)			
			*((uint16_t*)(buf->pkt_sync.hdr_sync_h2d  + offset + h2d_offset[i])) = (uint16_t)( (h2d_offset[i] == 10) ? dstPort : srcPort );
	}
	
	//数据包同步
	offset = pkt_idx * MAX_PKT_LEN;
	for(size_t i = 0 ; i < pld_size && pkt_sync_global.h2d_payload_flag ; i++)
		*((char*)(buf->pkt_sync.pld_sync_h2d + offset + i)) = *((char*)(pld_start + i));
	/*
	struct ipv4_hdr* hdr=onvm_pkt_ipv4_hdr(pkt);
	*((uint32_t*)(buf->gpu_sync + offset)) = hdr->src_addr;
	*((uint32_t*)(buf->gpu_sync + offset +4)) = hdr->dst_addr;
	*/
}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	/* Write the port */
	pkt->port = buf->host_out[pkt_idx];
	int pld_size = 0;
	char* pld_start = onvm_pkt_payload(pkt , &pld_size);

	//这里面套个switch就结束了
	//在d2h后，数据就已经从device获取到了
	uint32_t offset=pkt_idx*SYNC_DATA_SIZE;

	struct ipv4_hdr* ip_h=onvm_pkt_ipv4_hdr(pkt);
	struct tcp_hdr* tcp_h=onvm_pkt_tcp_hdr(pkt);
	struct udp_hdr* udp_h=onvm_pkt_udp_hdr(pkt);

	uint16_t* srcPort = NULL;
	uint16_t* dstPort = NULL;

	if(tcp_h != NULL)
	{
		srcPort = &tcp_h->src_port;
		dstPort = &tcp_h->dst_port;
	}
	else
	{
		srcPort = &udp_h->src_port;
		dstPort = &udp_h->dst_port;
	}
	

	//根据d2h_hint，进行遍历，将其中每个位的数据解析出来
	//下面这个代码是有点问题的 会seg fault
	for(size_t i = 0 ; i < pkt_sync_global.d2h_sync_num ; i++)
	{
		if(d2h_offset[i] <= 4)
		{
			
			if(d2h_offset[i] == 4)
				ip_h->dst_addr = *((uint32_t*)(buf->pkt_sync.hdr_sync_d2h + offset + d2h_offset[i]));
			else
				ip_h->src_addr = *((uint32_t*)(buf->pkt_sync.hdr_sync_d2h + offset + d2h_offset[i]));
		}
		else if(d2h_offset[i] <= 10)
		{
			if(d2h_offset[i] == 10)
				*dstPort = *((uint16_t*)(buf->pkt_sync.hdr_sync_d2h + offset + d2h_offset[i]));
			else
				*srcPort = *((uint16_t*)(buf->pkt_sync.hdr_sync_d2h + offset + d2h_offset[i]));
		}	
	}
		
	offset = pkt_idx * MAX_PKT_LEN;
	for(size_t i = 0;i < pld_size && pkt_sync_global.d2h_payload_flag ;i++)
		*((char*)(pld_start + i)) = *((char*)(buf->pkt_sync.pld_sync_d2h + offset + i));
	/*
	struct ipv4_hdr* hdr = onvm_pkt_ipv4_hdr(pkt);
	hdr->src_addr = *((uint32_t*)buf->cpu_sync);
	hdr->dst_addr = *((uint32_t*)buf->cpu_sync + 4);
	*/
}

static void user_gpu_htod(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->device_in, buf->host_in, job_num * sizeof(CUdeviceptr), ASYNC, thread_id);
	if(pkt_sync_global.h2d_sync_num!=0)
		gcudaMemcpyHtoD(buf->pkt_sync.d_hdr_sync_h2d, buf->pkt_sync.hdr_sync_h2d, job_num  * SYNC_DATA_SIZE * sizeof(uint8_t), ASYNC, thread_id);
	if(pkt_sync_global.h2d_payload_flag)
		gcudaMemcpyHtoD(buf->pkt_sync.d_pld_sync_h2d, buf->pkt_sync.pld_sync_h2d, job_num  * MAX_PKT_LEN * sizeof(uint8_t), ASYNC, thread_id);
}

static void user_gpu_dtoh(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_out, buf->device_out, job_num * sizeof(uint8_t), ASYNC, thread_id);
	if(pkt_sync_global.d2h_sync_num!=0)
		gcudaMemcpyDtoH(buf->pkt_sync.hdr_sync_d2h, buf->pkt_sync.d_hdr_sync_d2h, job_num  * SYNC_DATA_SIZE * sizeof(uint8_t), ASYNC, thread_id);
	if(pkt_sync_global.d2h_payload_flag)
		gcudaMemcpyDtoH(buf->pkt_sync.pld_sync_d2h, buf->pkt_sync.d_pld_sync_d2h, job_num  * MAX_PKT_LEN * sizeof(uint8_t), ASYNC, thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	//tmd 这个参数忘记改 花了我一晚上时间debug
	uint64_t arg_num = 15;
	uint64_t offset = 0;

	info[0] = arg_num;

	info[1] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->device_in), sizeof(buf->device_in));
	offset += sizeof(buf->device_in);

	info[2] = offset;
	//printf("count of data:%d\n",job_num);
	rte_memcpy((uint8_t *)arg_buf + offset, &(job_num), sizeof(job_num));
	offset += sizeof(job_num);
	
	info[3] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->device_out), sizeof(buf->device_out));
	offset += sizeof(buf->device_out);
	
	info[4] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(tbl24_d), sizeof(tbl24_d));
	offset += sizeof(tbl24_d);

	//同步相关参数
	//首部相关
	info[5] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &pkt_sync_global.h2d_sync_num, sizeof(pkt_sync_global.h2d_sync_num));
	offset += sizeof(pkt_sync_global.h2d_sync_num);

	info[6] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &pkt_sync_global.d2h_sync_num, sizeof(pkt_sync_global.d2h_sync_num));
	offset += sizeof(pkt_sync_global.d2h_sync_num);

	info[7] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->pkt_sync.d_hdr_sync_h2d) , sizeof(buf->pkt_sync.d_hdr_sync_h2d)) ;
	offset += sizeof(buf->pkt_sync.d_hdr_sync_h2d);

	info[8] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->pkt_sync.d_hdr_sync_d2h) , sizeof(buf->pkt_sync.d_hdr_sync_d2h)) ;
	offset += sizeof(buf->pkt_sync.d_hdr_sync_d2h);

	info[9] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(pkt_sync_global.h2d_offset_d) , sizeof(pkt_sync_global.h2d_offset_d));
	offset += sizeof(pkt_sync_global.h2d_offset_d);

	info[10] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(pkt_sync_global.d2h_offset_d) , sizeof(pkt_sync_global.d2h_offset_d));
	offset += sizeof(pkt_sync_global.d2h_offset_d);

	//针对数据包来说的同步参数
	info[11] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(pkt_sync_global.h2d_payload_flag) , sizeof(pkt_sync_global.h2d_payload_flag));
	offset += sizeof(pkt_sync_global.h2d_payload_flag);

	info[12] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(pkt_sync_global.d2h_payload_flag) , sizeof(pkt_sync_global.h2d_payload_flag));
	offset += sizeof(pkt_sync_global.d2h_payload_flag);

	info[13] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->pkt_sync.pld_sync_h2d) , sizeof(buf->pkt_sync.pld_sync_h2d));
	offset += sizeof(buf->pkt_sync.pld_sync_h2d);

	info[14] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->pkt_sync.pld_sync_d2h) , sizeof(buf->pkt_sync.pld_sync_d2h));
	offset += sizeof(buf->pkt_sync.pld_sync_d2h);

	info[15] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(pkt_sync_global.payload_size) , sizeof(pkt_sync_global.payload_size));
	offset += sizeof(pkt_sync_global.payload_size);
}

static void init_main(void)
{
	int table_item_num = 1 << 24;

	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * sizeof(CUdeviceptr)  // host_in
			+ MAX_BATCH_SIZE * sizeof(uint8_t) 			 // host_out
			+ MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(char) // h2d sync data
			+ MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(char) // d2h sync data
			+ MAX_BATCH_SIZE * MAX_PKT_LEN * sizeof(char) * 2 //用于同步数据包负载	 	
			,
			table_item_num * sizeof(uint16_t)			//全局用于同步的两个偏移量数组
			+ sizeof(uint16_t) * SYNC_DATA_COUNT * 2,
			0);                                       // first time

	gcudaMalloc(&tbl24_d, table_item_num * sizeof(uint16_t));
	gcudaMalloc(&pkt_sync_global.h2d_offset_d , sizeof(uint16_t) * SYNC_DATA_COUNT);
	gcudaMalloc(&pkt_sync_global.d2h_offset_d , sizeof(uint16_t) * SYNC_DATA_COUNT);

	gcudaHostAlloc((void **)&tbl24_h, table_item_num * sizeof(uint16_t));
	gcudaHostAlloc((void **)&h2d_offset , sizeof(uint16_t) * SYNC_DATA_COUNT);
	gcudaHostAlloc((void **)&d2h_offset , sizeof(uint16_t) * SYNC_DATA_COUNT);
	
	int i;
	for (i = 0; i < table_item_num; i ++) {
		tbl24_h[i] = i & (0xffff);
	}
	memset(h2d_offset,0,SYNC_DATA_COUNT);
	memset(d2h_offset,0,SYNC_DATA_COUNT);

	gcudaMemcpyHtoD(tbl24_d, tbl24_h, table_item_num * sizeof(uint16_t), SYNC, 0);
	gcudaMemcpyHtoD(pkt_sync_global.h2d_offset_d, h2d_offset, SYNC_DATA_COUNT * sizeof(uint16_t), SYNC, 0);
	gcudaMemcpyHtoD(pkt_sync_global.d2h_offset_d, d2h_offset, SYNC_DATA_COUNT * sizeof(uint16_t), SYNC, 0);
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
	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_ROUTER,GPU_NF,&(init_gpu_schedule))) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	//获取hint信息
	onvm_framework_get_hint(&h2d_hint , &d2h_hint , 
							h2d_offset, d2h_offset , 
							&pkt_sync_global);

	init_main();

	printf("EAL: h2d_sync_num:%d h2d_hint:%d \nEAL: d2h_sync_num:%d d2h_hint:%d\n",
			pkt_sync_global.h2d_sync_num,h2d_hint,
			pkt_sync_global.d2h_sync_num,d2h_hint);

	/* Initialization is done, start threads */
	onvm_framework_start_cpu(&(init_host_buf), &(user_batch_func), &(user_post_func),NULL,GPU_NF);

	onvm_framework_start_gpu(&(user_gpu_htod), &(user_gpu_dtoh), &(user_gpu_set_arg));

	printf("If we reach here, program is ending\n");
	return 0;
}
