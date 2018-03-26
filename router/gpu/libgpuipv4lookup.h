#ifndef IPV4LOOKUP_KERNEL_H
#define IPV4LOOKUP_KERNEL_H

void IPv4_Lookup(const uint32_t *input_buf,
			const uint32_t job_num,
			uint8_t *output_buf,
			const uint16_t *tbl24,
			const unsigned int threads_per_blk,
			const unsigned int num_cuda_blks,
			cudaStream_t stream);

#endif
