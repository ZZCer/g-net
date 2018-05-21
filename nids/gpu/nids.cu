#include <stdio.h>
#include <stdlib.h>
#include "rules.h"
#include <gpu_packet.h>

extern "C" __global__ void match(const uint16_t *acArray,
						gpu_packet_t **pkt_in,
						uint16_t *result,
						const int batch_size)
{
	uint32_t len;
	int i;
	uint16_t state = 0;
	uint8_t content;
	uint16_t mark;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int step = ceil(((float)batch_size) / (blockDim.x * gridDim.x));
	tid = tid * step;

	for (unsigned j = 0; (j < step) && (tid < batch_size); j++, tid++) {
		len = pkt_in[tid]->payload_size;

		mark = 0;

		for (i = 0; i < len; i ++) {
			content = pkt_in[tid]->payload[i];
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

