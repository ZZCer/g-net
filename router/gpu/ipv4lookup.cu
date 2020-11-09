#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <gpu_packet.h>

/*******************************************************************
  IPv4 Lookup with DIR-24-8-BASIC algorithm from Infocom'98 paper:
  <Routing Lookups in Hardware at Memory Access Speeds>
******************************************************************/
extern "C" __global__ void ipv4lookup(gpu_packet_t **input_buf,
			const int job_num,
			uint8_t *output_buf,
			const uint16_t *tbl24,
			uint8_t h2d_hint,
			uint8_t d2h_hint,
			char* sync_in,
			char* sync_out)
{
	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;
	int i;
	int j;
	uint32_t hash;
	uint16_t value_tb1;

	
	//根据hint再把同步数据拷贝给gpu
	uint8_t tp_hint=h2d_hint;
	//首先把hint中对应的五元组每一项下标保存到一个数组中
	//需要注意这个是逆序存储的，因此遍历的时候也需要逆序来遍历
	int sync_in_index[5];
	int sync_count=0;
	int sync_data_id=0;
	int tp_id=0;

	while(tp_hint!=0)
	{
		tp_id = sync_data_id++;
		if(tp_hint & 1 !=0)
		{		
			sync_in_index[tp_id]=5-sync_data_id;
			sync_count++;
		}
		tp_hint >>= 1;
	}

	//接下来进行h2d的同步
	for (i = idx; i < job_num; i += step)
	{
		for(j = sync_count-1;j>=0;j--)
		{
			switch(sync_in_index[j])
			{
				case SYNC_SOURCE_IP:
					input_buf[i]->src_addr = *((uint32_t*)(sync_in + i * SYNC_DATA_SIZE));
					break;
				case SYNC_DEST_IP:
					input_buf[i]->dst_addr = *((uint32_t*)(sync_in + i * SYNC_DATA_SIZE + 4));
					break;
				case SYNC_SOURCE_PORT:
					input_buf[i]->src_port = *((uint16_t*)(sync_in + i * SYNC_DATA_SIZE + 8));
					break;
				case SYNC_DEST_PORT:
					input_buf[i]->dst_port = *((uint16_t*)(sync_in + i * SYNC_DATA_SIZE + 10));
					break;
				case SYNC_TCP_FLAGS:
					input_buf[i]->tcp_flags = *((uint8_t*)(sync_in + i * SYNC_DATA_SIZE + 12));
					break;
			}
		}
	}

	//核心kernel的执行部分
	//先做h2d同步
	//这里是相当于并行化处理
	for (i = idx; i < job_num; i += step){
		hash = input_buf[i]->dst_addr >> 8;
		value_tb1 = tbl24[hash];
		output_buf[i] = (uint8_t)value_tb1; //FIXME
		//printf("in %x [%x - hash %x], v %x, uint8 %x\n", input_buf[i], i, hash, value_tb1, (uint8_t)value_tb1);
	}

	//核心算法完成后，进行d2h同步，h2d与d2h的同步可能是不相同的，因此需要重新求解下标项
	tp_hint = d2h_hint;
	sync_count = 0;
	sync_data_id = 0;
	tp_id = 0;

	//用同一个数组来存储同步下标项
	while(tp_hint!=0)
	{
		tp_id = sync_data_id++;
		if(tp_hint & 1 !=0)
		{		
			sync_in_index[tp_id]=5-sync_data_id;
			sync_count++;
		}
		tp_hint >>= 1;
	}

	for (i = idx; i < job_num; i += step)
	{
		for(j = sync_count-1;j>=0;j--)
		{
			switch(sync_in_index[j])
			{
				case SYNC_SOURCE_IP:
					*((uint32_t*)(sync_out + i * SYNC_DATA_SIZE)) = input_buf[i]->src_addr;
					break;
				case SYNC_DEST_IP:
					*((uint32_t*)(sync_out + i * SYNC_DATA_SIZE + 4)) = input_buf[i]->dst_addr;
					break;
				case SYNC_SOURCE_PORT:
					*((uint16_t*)(sync_out + i * SYNC_DATA_SIZE + 8)) = input_buf[i]->src_port;
					break;
				case SYNC_DEST_PORT:
					*((uint16_t*)(sync_out + i * SYNC_DATA_SIZE + 10)) = input_buf[i]->dst_port;
					break;
				case SYNC_TCP_FLAGS:
					*((uint8_t*)(sync_out + i * SYNC_DATA_SIZE + 12)) = input_buf[i]->tcp_flags;
					break;
			}
		}
	}
}

