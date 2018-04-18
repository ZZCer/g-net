#include <cuda.h>
#include <onvm_common.h>
#include <onvm_framework.h>

extern "C"
__global__ void load_packets(gpu_packet_t *data, gpu_packet_t **ptrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    *ptrs[i] = data[i];
}

extern "C"
__global__ void unload_packets(gpu_packet_t *data, gpu_packet_t **ptrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = *ptrs[i];
}
