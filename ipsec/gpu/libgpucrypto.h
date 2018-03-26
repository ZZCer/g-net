#ifndef CRYPTO_KERNEL_H
#define CRYPTO_KERNEL_H

#define SHA1_THREADS_PER_BLK 32
#define MAX_KEY_SIZE 64
#define MAX_HASH_SIZE 20

void co_aes_sha1_gpu(
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
			cudaStream_t stream);

void AES_cbc_128_decrypt_gpu(const uint8_t *in_d,
			     uint8_t *out_d,
			     uint8_t *keys_d,
			     uint8_t *ivs_d,
			     uint16_t *pkt_index_d,
			     unsigned long block_count,
			     uint8_t *checkbits_d,
			     const unsigned int threads_per_blk,
			     cudaStream_t stream);

void AES_cbc_128_encrypt_gpu(const uint8_t *in_d,
			     uint8_t *out_d,
			     const uint32_t* pkt_offset_d,
			     const uint8_t *keys_d,
			     uint8_t *ivs_d,
			     const unsigned int numFlows,
			     uint8_t *checkbits_d,
			     const unsigned int threads_per_blk,
			     const unsigned int num_cuda_blks,
			     cudaStream_t stream);

void AES_ecb_128_encrypt_gpu(const uint8_t *in_d,
			     uint8_t *out_d,
			     const uint8_t *keys_d,
			     uint16_t *pkt_index_d,
			     unsigned long block_count,
			     const unsigned int threads_per_blk,
			     cudaStream_t stream);


void AES_decrypt_key_prepare(uint8_t *dec_key,
			     const uint8_t *enc_key,
			     unsigned int key_bits);


void hmac_sha1_gpu(char *buf, char *keys,  uint32_t *offsets, uint16_t *lengths,
		   uint32_t *outputs, int N, uint8_t *checkbits,
		   unsigned threads_per_blk, cudaStream_t stream);

#endif /* CRYPTO_KERNEL_H */
