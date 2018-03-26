#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "sha1.h"

__global__ void computeHMAC_SHA1(char *buf, char *keys,  uint32_t *offsets, uint16_t *lengths, uint32_t *outputs, int N, uint8_t *checkbits)
{
	uint32_t w_register[16];

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		uint32_t *w = w_register;
		hash_digest_t h;
		uint32_t offset = offsets[index];
		uint16_t length = lengths[index];
		uint32_t *out = outputs + 5 * index;


		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x36363636;
		xorpads(w, (uint32_t*)(keys + 64 * index));


		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		//SHA1 compute on ipad
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA1 compute on mesage
		unsigned num_iter = (length + 63 + 9) / 64;
		for (unsigned i = 0; i < num_iter; i++)
			computeSHA1Block(buf + offset , w, i * 64  , length , h);

		*(out)   = swap(h.h1);
		*(out+1) = swap(h.h2);
		*(out+2) = swap(h.h3);
		*(out+3) = swap(h.h4);
		*(out+4) = swap(h.h5);

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x5c5c5c5c;

		xorpads(w, (uint32_t*)(keys + 64 * index));

		//SHA 1 compute on opads
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA 1 compute on (hash of ipad|m)
		computeSHA1Block((char*)out, w, 0, 20, h);

		*(out)   = swap(h.h1);
		*(out+1) = swap(h.h2);
		*(out+2) = swap(h.h3);
		*(out+3) = swap(h.h4);
		*(out+4) = swap(h.h5);
	}
        __syncthreads();

	if (threadIdx.x == 0)
		*(checkbits + blockIdx.x) = 1;

}

extern "C" void hmac_sha1_gpu(char *buf, char *keys,  uint32_t *offsets, uint16_t *lengths,
		   uint32_t *outputs, int N, uint8_t *checkbits,
		   unsigned threads_per_blk, cudaStream_t stream)
{
	int num_blks = (N + threads_per_blk - 1) / threads_per_blk;
	if (stream == 0) {
		computeHMAC_SHA1<<<num_blks, threads_per_blk>>>(
		       buf, keys, offsets, lengths, outputs, N, checkbits);
	} else  {
		computeHMAC_SHA1<<<num_blks, threads_per_blk, 0, stream>>>(
		       buf, keys, offsets, lengths, outputs, N, checkbits);
	}
}

