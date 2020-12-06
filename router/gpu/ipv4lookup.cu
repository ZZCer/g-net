#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gpu_packet_sync.h>

/*******************************************************************
  IPv4 Lookup with DIR-24-8-BASIC algorithm from Infocom'98 paper:
  <Routing Lookups in Hardware at Memory Access Speeds>
******************************************************************/
extern "C" __global__ void ipv4lookup(gpu_packet_t **input_buf,
			const int job_num,
			uint8_t *output_buf,
			const uint16_t *tbl24,
			//首部同步用参数
			uint16_t h2d_num,
			uint16_t d2h_num,
			char* sync_in,
			char* sync_out,
			uint16_t* h2d_offset,
			uint16_t* d2h_offset,
			//数据包同步用参数
			uint8_t h2d_pld_flag,
			uint8_t d2h_pld_flag,
			char* h2d_pld,
			char* d2h_pld,
			int payload_size
		)
{
	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;
	uint32_t hash = 0;
	uint16_t value_tb1 = 0;
	int i = 0;

	//SYNC_DATA_SIZE
	//从cuda内存地址中取数据必须内存对齐，或者说必须是2^n的内层大小来取数据（内存对齐：数据地址必须是数据大小的倍数）
	//比如直接从sync_in + 13*i 来取数据uint32_t，内存地址就不是其数据大小的倍数了
	//因此先把每个数据包对应的13字节数组导入到一个字节数组中，按字节传递（此时数据大小是1，随便什么地址都能够传）
	//接着再进行整数倍偏移，此时数据的地址就一定是它大小的倍数了	
	//sync_h2d_header(h2d_num,h2d_offset,input_buf,sync_in,idx,step,job_num);
	//sync_h2d_payload(h2d_pld_flag,input_buf,h2d_pld,idx,step,job_num,payload_size);
	
	//核心kernel的执行部分
	//先做h2d同步
	//这里是相当于并行化处理

	/*
	for (i = idx; i < job_num; i += step){
		hash = input_buf[i]->dst_addr >> 8;
		value_tb1 = tbl24[hash];
		output_buf[i] = (uint8_t)value_tb1; //FIXME
		//printf("in %x [%x - hash %x], v %x, uint8 %x\n", input_buf[i], i, hash, value_tb1, (uint8_t)value_tb1);
	}
	*/

	//sync_d2h_header(d2h_num,d2h_offset,input_buf,sync_out,idx,step,job_num);
	//sync_d2h_payload(d2h_pld_flag,input_buf,d2h_pld,idx,step,job_num,payload_size);
}

