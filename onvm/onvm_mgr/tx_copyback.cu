#include "../onvm_nflib/gpu_packet.h"

#define GPU_PKT_ALIGN 16
#define GPU_PKT_ALIGN_MASK (GPU_PKT_ALIGN - 1)
#define TX_GPU_KERN_SEG_SIZE 64

extern "C"
__global__ void tx_copyback(gpu_packet_t **gpkts, unsigned pktcnt, uint8_t **start_pos) {
    for (unsigned i = tid; i < pktcnt; i += step) {
        uint8_t *dst = start_pos[i];
        uint8_t *src = (uint8_t *)gpkts[i];
        unsigned copysize = sizeof(gpu_packet_t) + gpkts[i]->payload_size;
        memcpy(dst, src, copysize);
    }
}
