#include "../onvm_nflib/gpu_packet.h"

#define TX_GPU_BATCH_SIZE 4096
#define TX_GPU_BUF_SIZE (256*1024)
#define GPU_PKT_ALIGN 16
#define GPU_PKT_ALIGN_MASK (GPU_PKT_ALIGN - 1)
#define TX_GPU_KERN_SEG_SIZE 64

extern "C"
__global__ void tx_copyback(gpu_packet_t **gpkts, unsigned pktcnt, uint8_t *buffer, unsigned *unload_cnt) {
    __shared__ uint16_t buf_sz[TX_GPU_BATCH_SIZE];
    __shared__ unsigned start_off[TX_GPU_BATCH_SIZE];
    __shared__ unsigned seg_sz[TX_GPU_BATCH_SIZE / TX_GPU_KERN_SEG_SIZE];
    __shared__ unsigned max;
    const unsigned tid = threadIdx.x;
    const unsigned step = blockDim.x;

    const unsigned seglen = TX_GPU_KERN_SEG_SIZE;
    const unsigned segnum = (pktcnt + seglen - 1) / seglen;
    for (unsigned i = tid; i < pktcnt; i += step) {
        uint16_t bsz;
        bsz = sizeof(gpu_packet_t) + gpkts[i]->payload_size;
        bsz = (bsz + GPU_PKT_ALIGN_MASK) & (~GPU_PKT_ALIGN_MASK);
        buf_sz[i] = bsz;
    }
    __syncthreads();
    if (tid < segnum) {
        unsigned ssz = 0;
        for (int i = tid * seglen; i < (tid + 1) * seglen && i < pktcnt; i++) {
            ssz += buf_sz[i];
        }
        seg_sz[tid] = ssz;
    }
    __syncthreads();
    if (tid == 0) {
        unsigned cur_off = 0;
        unsigned cur_ssz = 0;
        int i;
        for (i = 0; i < segnum && cur_off < TX_GPU_BUF_SIZE; i++) {
            cur_ssz = seg_sz[i];
            seg_sz[i] = cur_off;
            cur_off += cur_ssz;
        }
        max = i;
        *unload_cnt = i * seglen;
    }
    __syncthreads();
    if (tid < segnum && tid < max) {
        unsigned cur_off = seg_sz[tid];
        for (int i = tid * seglen; i < (tid + 1) * seglen && i < pktcnt; i++) {
            start_off[i] = cur_off;
            cur_off += buf_sz[i];
        }
    }
    __syncthreads();
    for (unsigned i = tid; i < pktcnt; i += step) {
        if (start_off[i] + buf_sz[i] > TX_GPU_BUF_SIZE) {
            if (start_off[i] <= TX_GPU_BUF_SIZE) *unload_cnt = i;
            break;
        }
        uint8_t *dst = buffer + start_off[i];
        uint8_t *src = (uint8_t *)gpkts[i];
        unsigned copysize = buf_sz[i];
        memcpy(dst, src, copysize);
    }
}

