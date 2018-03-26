#ifndef LIBGPUIDS_H
#define LIBGPUIDS_H

void gpumatch(const int block_num, const int thread_per_block, const int batch_size, const uint16_t *acGPU, char *pkt, uint32_t *pkt_offset, uint16_t *res);

void test(ListRoot *tmp);
void freeall(ListRoot *tmp);

#endif
