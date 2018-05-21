#include <stdio.h>
#include <stdlib.h>
#include "rules.h"

extern "C" __global__ void match(const uint16_t *acArray,
						char *pkt_in,
						const uint32_t *pkt_offset,
						uint16_t *result,
						const int batch_size)
{
	uint32_t len, start;
	int i;
	uint16_t state = 0;
	uint8_t content;
	uint16_t mark;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int step = ceil(((float)batch_size) / (blockDim.x * gridDim.x));
	tid = tid * step;

	for (unsigned j = 0; (j < step) && (tid < batch_size); j++, tid++) {
		len = pkt_offset[tid + 1] - pkt_offset[tid];
		start = pkt_offset[tid];

		mark = 0;

		for (i = 0; i < len; i ++) {
			content = pkt_in[start + i];
			if (content == 0) continue; // we have to ignore 0 as the table use it to save results

			while (acArray[257 * state + content] == 0 && acArray[257 * state + 256] != 0) {
				state = acArray[257 * state + 256];
			}

			state = acArray[257 * state + content];

			/* FIXME: Only record the first matched pattern */
			if (mark == 0 && state != 0 && acArray[257 * state + 0] != 0) {
				mark = acArray[257 * state + 0];
				// no break to test performance
			}
		}

		result[tid] = mark;
	}
}

extern "C"
void gpumatch(const int block_num, const int thread_per_block, const int batch_size, const uint16_t *acGPU, char *pkt_in, const uint32_t *pkt_offset, uint16_t *res)
{
	match<<<block_num, thread_per_block>>>(acGPU, pkt_in, pkt_offset, res, batch_size);

	return;
}
