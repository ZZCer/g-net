#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <gpu_packet.h>

/*******************************************************************
  IPv4 Lookup with DIR-24-8-BASIC algorithm from Infocom'98 paper:
  <Routing Lookups in Hardware at Memory Access Speeds>
******************************************************************/
extern "C" __global__ void ipv4lookup(const gpu_packet_t **input_buf,
			const int job_num,
			uint8_t *output_buf,
			const uint16_t *tbl24)
{
	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;
	int i;
	uint32_t hash;
	uint16_t value_tb1;

	for (i = idx; i < job_num; i += step) {
		hash = input_buf[i]->dst_addr >> 8;
		value_tb1 = tbl24[hash];
		output_buf[i] = (uint8_t)value_tb1; //FIXME
		//printf("in %x [%x - hash %x], v %x, uint8 %x\n", input_buf[i], i, hash, value_tb1, (uint8_t)value_tb1);
	}
	return;
}

