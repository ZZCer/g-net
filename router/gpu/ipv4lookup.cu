#include <stdint.h>
#include <stdio.h>
#include <assert.h>

/*******************************************************************
  IPv4 Lookup with DIR-24-8-BASIC algorithm from Infocom'98 paper:
  <Routing Lookups in Hardware at Memory Access Speeds>
******************************************************************/
extern "C" __global__ void ipv4lookup(const uint32_t *input_buf,
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
		hash = input_buf[i] >> 8;
		value_tb1 = tbl24[hash];
		output_buf[i] = (uint8_t)value_tb1; //FIXME
		//printf("in %x [%x - hash %x], v %x, uint8 %x\n", input_buf[i], i, hash, value_tb1, (uint8_t)value_tb1);
	}
	return;
}


/**************************************************************************
 Exported C++ function wrapper function for CUDA kernel
***************************************************************************/
extern "C" void IPv4_Lookup(const uint32_t *input_buf,
			const uint32_t job_num,
			uint8_t *output_buf,
			const uint16_t *tbl24,
			const unsigned int threads_per_blk,
			const unsigned int num_cuda_blks,
			cudaStream_t stream)
{
	//printf("%d = %d\n", threads_per_blk, num_cuda_blks);
	if (stream == 0) {
		ipv4lookup<<<num_cuda_blks, threads_per_blk>>>(
		    input_buf, job_num, output_buf, tbl24); 
	} else {
		ipv4lookup<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
		    input_buf, job_num, output_buf, tbl24); 
	}
}

