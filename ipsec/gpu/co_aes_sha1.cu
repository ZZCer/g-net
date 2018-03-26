#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "aes_core.h"
#include "sha1.h"
#include "crypto_size.h"

/* AES counter mode + HMAC SHA-1, 
   the encryption of each block in AES counter mode is not parallelized in this implementation */
extern "C" __global__ void
aes_ctr_sha1_kernel(
			const uint8_t	*input_buf,
			uint8_t *output_buf,
			const uint32_t *pkt_offset,
			const uint16_t *length,
			const uint8_t *aes_keys,
			const uint8_t *hmac_keys,
			const unsigned int num_flows,
			uint8_t *checkbits)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int step = ceil(((float)num_flows) / (blockDim.x * gridDim.x));
	idx = idx * step;

/**************************************************************************
  AES Encryption is started first
 ***************************************************************************/
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];
	__shared__ uint32_t shared_Rcon[10];

	/* initialize T boxes */
	for (unsigned i = 0; i * blockDim.x < 256 ; i ++) {
		unsigned index = threadIdx.x + i * blockDim.x;
		if (index >= 256)
			break;
		shared_Te0[index] = Te0_ConstMem[index];
		shared_Te1[index] = Te1_ConstMem[index];
		shared_Te2[index] = Te2_ConstMem[index];
		shared_Te3[index] = Te3_ConstMem[index];
	}

	for(unsigned i = 0; i * blockDim.x < 10; i ++){
		int index = threadIdx.x + blockDim.x * i;
		if(index < 10){
			shared_Rcon[index] = rcon[index];
		}
	}

	/* ----debug-----*/
	if (idx >= num_flows) {
		return;
	}

	/* make sure T boxes have been initialized. */
	__syncthreads();

	uint16_t len;

	for (unsigned i = 0; (i < step) && (idx < num_flows); i++, idx++) {

		/* ============================== AES CTR =============================== */
		uint64_t counter[2] = {0, 0};

		/* Locate data */
		const uint8_t *in  = pkt_offset[idx] + input_buf;
		uint8_t *out       = pkt_offset[idx] + output_buf;
		const uint8_t *key = idx * AES_KEY_SIZE + aes_keys;

		/* Encrypt using cbc mode */
		len = length[idx];

		while (len >= AES_BLOCK_SIZE) {
			/* Update counter for each block */
			counter[0] ++;
			if (counter[0] == 0) counter[1] ++;

			AES_128_encrypt((uint8_t *)counter, out, key,
					shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);

			*((uint64_t*)out)       = *((uint64_t*)in)       ^ *((uint64_t*)out);
			*(((uint64_t*)out) + 1) = *(((uint64_t*)in) + 1) ^ *(((uint64_t*)out) + 1);

			len -= AES_BLOCK_SIZE;
			in  += AES_BLOCK_SIZE;
			out += AES_BLOCK_SIZE;
		}


		if (len) {
			counter[0] ++;
			if (counter[0] == 0) counter[1] ++;

			AES_128_encrypt((uint8_t *)counter, out, key,
					shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);

			for(unsigned n = 0; n < len; ++n)
				out[n] = in[n] ^ out[n];
		}

		__syncthreads();

		/**************************************************************************
		  AES Encryption completed, Now we start SHA-1 Calculation
		 ***************************************************************************/
		uint32_t w_register[16];

		uint32_t *w = w_register;
		hash_digest_t h;
		uint32_t offset = pkt_offset[idx];
		len = length[idx];
		uint16_t sha1_output_pos = (len + 3) & ~0x03;
		//printf("len %d, pad_len %d, sha1_output_pos %d\n", len, pkt_offset[idx+1]-pkt_offset[idx], sha1_output_pos);
		if (sha1_output_pos > pkt_offset[idx+1] - pkt_offset[idx] - HMAC_TAG_SIZE) {
			printf("ERROR: Sha1 output position exceeds the packet boundary! len %d, output %d, pad_len %d, hmac %d\n",
					len, sha1_output_pos, pkt_offset[idx+1] - pkt_offset[idx], HMAC_TAG_SIZE);
			return;
		}
		uint32_t *sha1_out = (uint32_t *)(input_buf + offset + sha1_output_pos);

		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x36363636;
		xorpads(w, (uint32_t *)(hmac_keys + HMAC_KEY_SIZE * idx));

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		//SHA1 compute on ipad
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA1 compute on message
		unsigned num_iter = (pkt_offset[idx+1] - pkt_offset[idx]) / 64;
		for (unsigned i = 0; i < num_iter; i ++)
			computeSHA1Block((char *)(input_buf + offset), w, i * 64, len, h);

		/* In SRTP, sha1_out has only 80 bits output 32+32+16 = 80 */
		*(sha1_out)   = swap(h.h1);
		*(sha1_out+1) = swap(h.h2);
		*(sha1_out+2) = swap(h.h3);
		*(sha1_out+3) = swap(h.h4);
		*(sha1_out+4) = swap(h.h5);

		h.h1 = 0x67452301;
		h.h2 = 0xEFCDAB89;
		h.h3 = 0x98BADCFE;
		h.h4 = 0x10325476;
		h.h5 = 0xC3D2E1F0;

		for (unsigned i = 0; i < 16; i++)
			w[i] = 0x5c5c5c5c;

		xorpads(w, (uint32_t*)(hmac_keys + HMAC_KEY_SIZE * idx));

		//SHA 1 compute on opads
		computeSHA1Block((char*)w, w, 0, 64, h);

		//SHA 1 compute on (hash of ipad|m)
		computeSHA1Block((char*)sha1_out, w, 0, 20, h);

		*(sha1_out)   = swap(h.h1);
		*(sha1_out+1) = swap(h.h2);
		*(sha1_out+2) = swap(h.h3);
		*(sha1_out+3) = swap(h.h4);
		*(sha1_out+4) = swap(h.h5);

		__syncthreads();
	}
	return;
}

extern "C" void co_aes_sha1_gpu(
			const uint8_t		*in,
			uint8_t				*out,
			const uint32_t		*pkt_offset,
			const uint16_t		*actual_length,
			const uint8_t		*aes_keys,
			const uint8_t		*hmac_keys,
			const unsigned int 	num_flows,
			uint8_t				*checkbits,
			const unsigned int	threads_per_blk,
			const unsigned int	num_blks,
			cudaStream_t stream)
{
	//printf("stream=%d, threads_per_blk =%d, num_blks = %d\n", stream, threads_per_blk, num_blks);
	if (stream == 0) {
		aes_ctr_sha1_kernel<<<num_blks, threads_per_blk>>>(
		       in, out, pkt_offset, actual_length, aes_keys, hmac_keys, num_flows, checkbits);
	} else  {
		aes_ctr_sha1_kernel<<<num_blks, threads_per_blk, 0, stream>>>(
		       in, out, pkt_offset, actual_length, aes_keys, hmac_keys, num_flows, checkbits);
	}
}

