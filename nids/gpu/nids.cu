#include <stdio.h>
#include <stdlib.h>
#include "rules.h"

extern "C" __global__ void match(const uint16_t *acArray,
						char *pkt_in,
						const uint32_t *pkt_offset,
						uint16_t *result,
						const int batch_size)
{
	uint32_t len;
	int i, state = 0;
	char content;
	int start, mark;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int step = ceil(((float)batch_size) / (blockDim.x * gridDim.x));
	tid = tid * step;

	for (unsigned j = 0; (j < step) && (tid < batch_size); j++, tid++) {
		len = pkt_offset[tid + 1] - pkt_offset[tid];
		start = pkt_offset[tid];

		mark = 0;

		for (i = 0; i < len; i ++) {
			content = pkt_in[start + i];

#if 1
			while (acArray[257 * state + ((int)content - 0)] == 0 && acArray[257 * state + 256] != 0) {
				state = acArray[257 * state + 256];
			}

			state = acArray[257 * state + ((int)content - 0)];

			/* FIXME: Only record the first matched pattern */
			if (mark == 0 && state != 0 && acArray[257 * state + 0] != 0) {
				result[tid] = acArray[257 * state + 0];
				mark = 1;
			}
#else
			int tmp;
			while (1) {
				tmp = acArray[257 * state + ((int)content - 0)];
				if (tmp != 0) {
					if (acArray[257 * tmp + 0] != 0) {
						/* FIXME: only record the first identified pattern */
						result[tid] = acArray[257 * tmp + 0];
					}
					state = tmp;
					break;
				} else {
					/* current state goes to nowhere with the character "content" */
					if (state == 0) break;
					else state = acArray[257 * state + 256]; /* this state fails, which state to go */
				}
			}
#endif
		}
	}
}

extern "C"
void gpumatch(const int block_num, const int thread_per_block, const int batch_size, const uint16_t *acGPU, char *pkt_in, const uint32_t *pkt_offset, uint16_t *res)
{
	match<<<block_num, thread_per_block>>>(acGPU, pkt_in, pkt_offset, res, batch_size);

	return;
}
