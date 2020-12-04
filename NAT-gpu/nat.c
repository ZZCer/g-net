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

#define NF_TAG "nat"
#define MAX_SIZE_PORTS 65535
#define SIZE_IP 4 //下述单位是字节
#define SIZE_PORT 2

//记录同步数据情况的全局变量
static uint8_t h2d_hint;
static uint8_t d2h_hint; 
uint16_t* h2d_offset;
uint16_t* d2h_offset;
CUdeviceptr h2d_offset_d;
CUdeviceptr d2h_offset_d;
uint16_t h2d_sync_num;
uint16_t d2h_sync_num;

//nat需要的全局变量端口到端口的映射
//端口到ip的映射
//端口集合
//内网ip，外网ip
uint32_t InIp;
uint32_t OutIp;

static uint16_t *Port2Port;
static uint32_t *Port2Ip;
static char *PortSet;
static uint32_t *InternalIp;
static uint32_t *ExternalIp;
static uint32_t *Mask;
static uint8_t* IPSet; 

static CUdeviceptr devPort2Port;
static CUdeviceptr devPort2Ip;
static CUdeviceptr devPortSet;
static CUdeviceptr devPort2Port;
static CUdeviceptr devInternalIp;
static CUdeviceptr devExternalIp;
static CUdeviceptr devMask;
static CUdeviceptr devIPSet;


typedef struct my_buf_s {
	/* Stores real data */
	CUdeviceptr *host_in;
	uint8_t *host_out;
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
	gcudaHostAlloc((void **)&(buf->gpu_sync), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	gcudaHostAlloc((void **)&(buf->cpu_sync), MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	
	gcudaMalloc(&(buf->device_in), MAX_BATCH_SIZE * sizeof(CUdeviceptr));
	gcudaMalloc(&(buf->device_out), MAX_BATCH_SIZE * sizeof(uint8_t));
	gcudaMalloc(&(buf->device_sync_in),MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	gcudaMalloc(&(buf->device_sync_out),MAX_BATCH_SIZE * SYNC_DATA_SIZE * sizeof(uint8_t));
	
	return buf;
}

static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	buf->host_in[pkt_idx] = onvm_pkt_gpu_ptr(pkt);

	uint32_t offset = pkt_idx * SYNC_DATA_SIZE;
	//printf("PKT_IDX:%d offset:%d\n",pkt_idx,offset);

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
	
	for(size_t i=0 ; i < h2d_sync_num ; i++)
	{
		if(h2d_offset[i] <= 4)			
			*((uint32_t*)(buf->gpu_sync + offset + h2d_offset[i])) = ( (h2d_offset[i] == 4) ? ip_h->dst_addr : ip_h->src_addr );
		else if(h2d_offset[i] <= 10)			
			*((uint16_t*)(buf->gpu_sync + offset + h2d_offset[i])) = (uint16_t)( (h2d_offset[i] == 10) ? dstPort : srcPort );
	}
}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

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
	for(size_t i = 0 ; i < d2h_sync_num ; i++)
	{
		if(d2h_offset[i] <= 4)
		{
			
			if(d2h_offset[i] == 4)
				ip_h->dst_addr = *((uint32_t*)(buf->cpu_sync + offset + d2h_offset[i]));
			else
				ip_h->src_addr = *((uint32_t*)(buf->cpu_sync + offset + d2h_offset[i]));
		}
		else if(d2h_offset[i] <= 10)
		{
			if(d2h_offset[i] == 10)
				*dstPort = *((uint16_t*)(buf->cpu_sync + offset + d2h_offset[i]));
			else
				*srcPort = *((uint16_t*)(buf->cpu_sync + offset + d2h_offset[i]));
		}	
	}

	if(buf->host_out[pkt_idx] == 0)
	{
		struct onvm_pkt_meta *meta;
		meta = onvm_get_pkt_meta((struct rte_mbuf *)pkt);
		meta->action = ONVM_NF_ACTION_DROP;
	}
}

static void user_gpu_htod(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->device_in, buf->host_in, job_num * sizeof(CUdeviceptr), ASYNC, thread_id);
	if(h2d_sync_num != 0)
		gcudaMemcpyHtoD(buf->device_sync_in, buf->gpu_sync , job_num * SYNC_DATA_SIZE * sizeof(uint8_t) , ASYNC , thread_id);
}

static void user_gpu_dtoh(void *cur_buf, int job_num, unsigned int thread_id)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_out, buf->device_out, job_num * sizeof(uint8_t), ASYNC, thread_id);
	if(d2h_sync_num!=0)
		gcudaMemcpyDtoH(buf->cpu_sync, buf->device_sync_out, job_num * SYNC_DATA_SIZE * sizeof(uint8_t), ASYNC, thread_id);
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	uint64_t arg_num = 16;
	uint64_t offset = 0;

	info[0] = arg_num;

	info[1] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->device_in), sizeof(buf->device_in));
	offset += sizeof(buf->device_in);

	info[2] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(job_num), sizeof(job_num));
	offset += sizeof(job_num);
	
	info[3] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devInternalIp), sizeof(devInternalIp));
	offset += sizeof(devInternalIp);

    info[4] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devExternalIp), sizeof(devExternalIp));
	offset += sizeof(devExternalIp);

    info[5] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devPort2Port), sizeof(devPort2Port));
	offset += sizeof(devPort2Port);

    info[6] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devPort2Ip), sizeof(devPort2Ip));
	offset += sizeof(devPort2Ip);

    info[7] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devPortSet), sizeof(devPortSet));
	offset += sizeof(devPortSet);

    info[8] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devMask), sizeof(devMask));
	offset += sizeof(devMask);

    info[9] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->device_out), sizeof(buf->device_out));
	offset += sizeof(buf->device_out);

	info[10] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(devIPSet), sizeof(devIPSet));
	offset += sizeof(devIPSet);

	//同步相关参数
	info[11] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &h2d_sync_num, sizeof(h2d_sync_num));
	offset += sizeof(h2d_sync_num);

	info[12] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &d2h_sync_num, sizeof(d2h_sync_num));
	offset += sizeof(d2h_sync_num);

	info[13] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->device_sync_in) , sizeof(buf->device_sync_in)) ;
	offset += sizeof(buf->device_sync_in);

	info[14] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(buf->device_sync_out) , sizeof(buf->device_sync_out)) ;
	offset += sizeof(buf->device_sync_out);

	info[15] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(h2d_offset_d) , sizeof(h2d_offset_d));
	offset += sizeof(h2d_offset_d);

	info[16] = offset;
	rte_memcpy((uint8_t*)arg_buf + offset, &(d2h_offset_d) , sizeof(d2h_offset_d));
	offset += sizeof(d2h_offset_d);
}

static void init_main(void)
{
	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * sizeof(CUdeviceptr)  // host_in
			+ MAX_BATCH_SIZE * sizeof(uint8_t) 
			+ sizeof(uint8_t) * MAX_BATCH_SIZE * SYNC_DATA_SIZE * 2,                              // host_out ，每个数据包输出40数据，前32位ip地址，后8位中的后4位是端口
			MAX_SIZE_PORTS * SIZE_PORT+                   //端口到端口映射  
            MAX_SIZE_PORTS * SIZE_IP+                     //端口到ip映射
            MAX_SIZE_PORTS * sizeof(char)+
            3*sizeof(uint32_t)+							// 内外网端口加上子网掩码
			UINT16_MAX,	
			0);                                       // first time

    //host全局变量
	gcudaHostAlloc((void **)&Port2Port, MAX_SIZE_PORTS * SIZE_PORT);
    gcudaHostAlloc((void **)&Port2Ip, MAX_SIZE_PORTS * SIZE_IP);
	gcudaHostAlloc((void **)&PortSet, MAX_SIZE_PORTS * sizeof(char));
	gcudaHostAlloc((void **)&InternalIp, SIZE_IP);
	gcudaHostAlloc((void **)&ExternalIp, SIZE_IP);
    gcudaHostAlloc((void **)&Mask,SIZE_IP);
	gcudaHostAlloc((void **)&IPSet,UINT16_MAX);
	gcudaHostAlloc((void **)&h2d_offset,sizeof(uint16_t) * SYNC_DATA_COUNT);
	gcudaHostAlloc((void **)&d2h_offset,sizeof(uint16_t) * SYNC_DATA_COUNT);

    //cuda内存分配
    gcudaMalloc(&devPort2Port,  MAX_SIZE_PORTS * SIZE_PORT);
    gcudaMalloc(&devPort2Ip,  MAX_SIZE_PORTS * SIZE_IP);
    gcudaMalloc(&devPortSet,  MAX_SIZE_PORTS * sizeof(char));
    gcudaMalloc(&devInternalIp,  SIZE_IP);
    gcudaMalloc(&devExternalIp,  SIZE_IP);
    gcudaMalloc(&devMask,  SIZE_IP);
	gcudaMalloc(&devIPSet,UINT16_MAX);
	gcudaMalloc(&h2d_offset_d , sizeof(uint16_t) * SYNC_DATA_COUNT);
	gcudaMalloc(&d2h_offset_d , sizeof(uint16_t) * SYNC_DATA_COUNT);

    //host全局变量初始化
	(*InternalIp)=InIp;
    (*ExternalIp)=OutIp;
    (*Mask)=IPv4(255,255,0,0);
	memset(IPSet,0,UINT16_MAX);
	memset(h2d_offset,0,SYNC_DATA_COUNT);
	memset(d2h_offset,0,SYNC_DATA_COUNT);
	for (int i = 0; i < MAX_SIZE_PORTS; i ++) {
		Port2Port[i]=0;
        Port2Ip[i]=0;
        PortSet[i]=0;
	}

    //将全部变量拷贝给cuda
	gcudaMemcpyHtoD(devPort2Port, Port2Port, MAX_SIZE_PORTS * SIZE_PORT, SYNC, 0);
	gcudaMemcpyHtoD(devPort2Ip, Port2Ip, MAX_SIZE_PORTS * SIZE_IP, SYNC, 0);
	gcudaMemcpyHtoD(devPortSet, PortSet, MAX_SIZE_PORTS * sizeof(char), SYNC, 0);
	gcudaMemcpyHtoD(devInternalIp, InternalIp, SIZE_IP, SYNC, 0);
	gcudaMemcpyHtoD(devExternalIp, ExternalIp, SIZE_IP, SYNC, 0);
	gcudaMemcpyHtoD(devMask, Mask, SIZE_IP, SYNC, 0);
	gcudaMemcpyHtoD(devIPSet, IPSet, SIZE_IP, SYNC, 0);

	gcudaMemcpyHtoD(h2d_offset_d, h2d_offset, SYNC_DATA_COUNT * sizeof(uint16_t), SYNC, 0);
	gcudaMemcpyHtoD(d2h_offset_d, d2h_offset, SYNC_DATA_COUNT * sizeof(uint16_t), SYNC, 0);
}

static void init_gpu_schedule(void)
{
	/* Initialize the GPU info, onvm_framework_init should be performed before onvm_nflib_init */
	const char *module_file = "../NAT-gpu/gpu/nat.ptx";
	const char *kernel_name = "nat";
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
	InIp=IPv4(192,168,0,0);
	OutIp=IPv4(10,176,64,36);

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_NAT,GPU_NF,(&init_gpu_schedule))) < 0)
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
