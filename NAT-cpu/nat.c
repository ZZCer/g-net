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
#include <emmintrin.h>
#include <signal.h>

#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_ether.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_framework.h"

#define NF_TAG "NAT"
#define MAX_SIZE_PORTS 65535
#define SIZE_IP 4 //下述单位是字节
#define SIZE_PORT 2

//nat需要的全局变量端口到端口的映射
//端口到ip的映射
//端口集合
//内网ip，外网ip
static uint32_t InIp;
static uint32_t OutIp;
static uint32_t Mask;

static uint16_t *Port2Port;
static uint32_t *Port2Ip;
static char *PortSet;

//对于CPU来说，只要计算就完事了，不需要考虑同步的问题
static uint8_t GR = 0;
static uint8_t GW = 0;
static uint8_t CR = 0b11110000;
static uint8_t CW = 0b11110000;

//记录了每个数据的偏移量
uint16_t sync_offset[SYNC_DATA_COUNT]={4,4,2,2,1};

//一些常用的数据量
#define Ether_Hdr_Len 14
#define IP_Hdr_Len 20
#define TCP_Hdr_Len 20

static char IPSet[UINT16_MAX];

static inline void cpu_handle(struct rte_mbuf *data)
{	
	u_char* pkt=rte_pktmbuf_mtod(data,u_char*);

	//数据信息
	uint32_t SrcIP;
	uint32_t DesIP;
	uint32_t SrcPort;
	uint32_t DesPort;
	//位置信息
	u_char *SrcIP_ptr;
	u_char *DesIP_ptr;
	u_char *SrcPort_ptr;
	u_char *DesPort_ptr;

	u_char* IP_Hdr=pkt+Ether_Hdr_Len;
	IP_Hdr+=12;
	SrcIP_ptr=IP_Hdr;
	SrcIP=*((uint32_t*)IP_Hdr);
	IP_Hdr+=4;
	DesIP_ptr=IP_Hdr;
	DesIP=*((uint32_t*)IP_Hdr);

	u_char* TCP_Hdr=pkt+Ether_Hdr_Len+IP_Hdr_Len;
	SrcPort_ptr=TCP_Hdr;
	SrcPort=*((uint16_t*)TCP_Hdr);
	TCP_Hdr+=2;
	DesPort_ptr=TCP_Hdr;
	DesPort=*((uint16_t*)TCP_Hdr);
	
	int DropTag=0;

	if((InIp & Mask)==( SrcIP & Mask))
	{
		uint16_t src=SrcPort;
		//内网
		if(IPSet[(uint16_t)(SrcIP & 0xFFFF)]==1)
			src=Port2Port[src];
		else
		{
			while(PortSet[src]!=0)
				src=(src+1)%65535;

			PortSet[src]=1;
			IPSet[(uint16_t)(SrcIP & 0xFFFF)]=1;
			Port2Port[src]=SrcPort;
			Port2Ip[src]=SrcIP;
		}	
		
		(*SrcIP_ptr)=OutIp;
		(*SrcPort_ptr)=src;
	}
	else
	{
		//外网
		//被端口号过滤掉了
		if(PortSet[DesPort]==0||DesIP!=OutIp)
			DropTag=1;
		else
		{
			uint16_t dst=Port2Port[DesPort];
			uint32_t dstip0=Port2Ip[DesPort];

			Port2Ip[DesPort]=0;
			Port2Port[DesPort]=0;
			IPSet[(uint16_t)(dstip0 & 0xFFFF)]=0;
			PortSet[DesPort]=0;

			(*DesIP_ptr)=dstip0;
			(*DesPort_ptr)=dst;
		}
	}

	if(DropTag != 0)
	{
		struct onvm_pkt_meta *meta;
		meta = onvm_get_pkt_meta((struct rte_mbuf *)pkt);
		meta->action = ONVM_NF_ACTION_DROP;
	}
}

static void init_main(void)
{
	InIp=IPv4(192,168,0,0);
	OutIp=IPv4(10,176,64,36);
	Mask=IPv4(255,255,0,0);

	Port2Port=malloc(sizeof(uint16_t)*MAX_SIZE_PORTS);
	Port2Ip=malloc(sizeof(uint32_t)*MAX_SIZE_PORTS);
	PortSet=malloc(sizeof(char)*MAX_SIZE_PORTS);

	memset(IPSet,0,UINT16_MAX);

	for(int i=0;i<MAX_SIZE_PORTS;i++)
	{
		Port2Port[i]=0;
		Port2Ip[i]=0;
		PortSet[i]=0;
	}

	printf("global data init\n");
}

static void free_main(void)
{
	if(Port2Ip!=NULL)
		free(Port2Ip);
	if(PortSet!=NULL)
		free(PortSet);
	if(Port2Port!=NULL)
		free(Port2Port);
}

int main(int argc, char *argv[])
{
	int arg_offset;

	hints hint={
		.CR=CR,
		.CW=CW,
		.GR=GR,
		.GW=GW
	};

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, hint ,NF_TAG, NF_NAT,CPU_NF ,NULL)) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	// 此时这里应该只需要分配host数据就可以了
	init_main();
	
	/* Initialization is done, start threads */
	onvm_framework_start_cpu(NULL,NULL, NULL,(&cpu_handle),CPU_NF);

	onvm_framework_cpu_only_wait();

	printf("start free nf global mem\n");
	free_main();
	
	printf("If we reach here, program is ending\n");
	return 0;
}
