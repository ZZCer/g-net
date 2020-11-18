/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2016 George Washington University
 *            2015-2016 University of California Riverside
 *            2010-2014 Intel Corporation. All rights reserved.
 *            2016 Hewlett Packard Enterprise Development LP
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * The name of the author may not be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************/


/******************************************************************************
  main.c

  File containing the main function of the manager and all its worker
  threads.

 ******************************************************************************/

// CLEAR: 1

#include "onvm_mgr.h"
#include "onvm_nf.h"
#include "onvm_stats.h"
#include "onvm_init.h"
#include "manager.h"
#include "scheduler.h"
#include "onvm_pkt_helper.h"

#include <rte_memory.h>

// Test flags to test pure RX performance
// define DROP_RX_PKTS
// #define DISABLE_GPU_RX
// #define DISABLE_TX
 #define ENABLE_PSTACK

extern struct onvm_service_chain *default_chain;
extern struct rx_perf rx_stats[ONVM_NUM_RX_THREADS]; 
extern struct port_info *ports;
extern CUdeviceptr gpu_pkts_buf;
extern CUdeviceptr gpu_pkts_head;
extern volatile CUdeviceptr gpu_pkts_tail;
extern CUstream rx_stream;
extern rte_spinlock_t gpu_pkts_lock;
extern CUcontext context;

extern uint8_t last_plan;

#define RX_BUF_SIZE (1024*64)
#define RX_BUF_PKT_MAX_NUM (RX_BUF_SIZE / 32)
#define RX_NUM_THREADS ONVM_NUM_RX_THREADS
#define RX_NUM_BATCHES 4

//rx对于从以太网接收到的数据是通过批来处理的，这是单个批的数据结构
typedef struct rx_batch_s {
    volatile int gpu_sync __rte_cache_aligned;
    volatile int full[RX_NUM_THREADS] __rte_cache_aligned;
    CUdeviceptr buf_head;
    unsigned pkt_cnt[RX_NUM_THREADS];
    struct rte_mbuf *pkt_ptr[RX_NUM_THREADS][RX_BUF_PKT_MAX_NUM];
    uint8_t buf[RX_NUM_THREADS][RX_BUF_SIZE];
#ifdef MEASURE_RX_LATENCY
    struct timespec batch_start_time[RX_NUM_THREADS];
#endif
} rx_batch_t;

static rx_batch_t rx_batch[RX_NUM_BATCHES];

//tx_batch中ready是针对每个端口号其threads来进行处理的
typedef struct tx_batch_s {
    volatile int ready[ONVM_NUM_TX_THREADS_PER_PORT];
    unsigned pkt_cnt;
    struct rte_mbuf **pkt_ptr;
    uint8_t *buf;
    CUdeviceptr buf_base;
#ifdef MEASURE_TX_LATENCY
    struct timespec batch_start_time[ONVM_NUM_TX_THREADS_PER_PORT];
#endif
} tx_batch_t;

static tx_batch_t tx_batch[ONVM_NUM_TX_THREADS_PER_PORT];
static __thread int current_tx_qid;

/*******************************Worker threads********************************/

static size_t size_packet(struct rte_mbuf *pkt) {
    struct ipv4_hdr* ipv4 = onvm_pkt_ipv4_hdr(pkt);
    if (!ipv4) {
        return sizeof(gpu_packet_t);
    }
    if (ipv4->next_proto_id == IPPROTO_TCP) {
        struct tcp_hdr *tcp = onvm_pkt_tcp_hdr(pkt);
        uint8_t *datastart, *dataend;
        datastart = (uint8_t *)tcp + (tcp->data_off >> 4 << 2);
        dataend = (uint8_t *)ipv4 + rte_be_to_cpu_16(ipv4->total_length);
        return sizeof(gpu_packet_t) + dataend - datastart;
    } else if (ipv4->next_proto_id == IPPROTO_UDP) {
        struct udp_hdr *udp = onvm_pkt_udp_hdr(pkt);
        return sizeof(gpu_packet_t) + rte_be_to_cpu_16(udp->dgram_len) - 8;
    } else {
        return sizeof(gpu_packet_t);
    }
}

static size_t load_packet(uint8_t *buffer, struct rte_mbuf *pkt) {
       gpu_packet_t *gpkt = (gpu_packet_t *) buffer;
       uint8_t *datastart, *dataend;
       struct ipv4_hdr* ipv4 = onvm_pkt_ipv4_hdr(pkt);
       if (!ipv4) {
               gpkt->proto_id = 0xFF;
               return sizeof(gpu_packet_t);
       }
       
       gpkt->src_addr = ipv4->src_addr;
       gpkt->dst_addr = ipv4->dst_addr;
       gpkt->proto_id = ipv4->next_proto_id;
       if (ipv4->next_proto_id == IPPROTO_TCP) {
               struct tcp_hdr *tcp = onvm_pkt_tcp_hdr(pkt);
               gpkt->tcp_flags = tcp->tcp_flags;
               gpkt->src_port = rte_be_to_cpu_16(tcp->src_port);
               gpkt->dst_port = rte_be_to_cpu_16(tcp->dst_port);
               gpkt->sent_seq = rte_be_to_cpu_32(tcp->sent_seq);
               gpkt->recv_ack = rte_be_to_cpu_32(tcp->recv_ack);
               datastart = (uint8_t *)tcp + (tcp->data_off >> 4 << 2);
               dataend = (uint8_t *)ipv4 + rte_be_to_cpu_16(ipv4->total_length);
               gpkt->payload_size = dataend - datastart;
       } else if (ipv4->next_proto_id == IPPROTO_UDP) {
               struct udp_hdr *udp = onvm_pkt_udp_hdr(pkt);
               gpkt->src_port = rte_be_to_cpu_16(udp->src_port);
               gpkt->dst_port = rte_be_to_cpu_16(udp->dst_port);
               datastart = (uint8_t *)udp + 8;
               gpkt->payload_size = rte_be_to_cpu_16(udp->dgram_len) - 8;
       } else {
               gpkt->payload_size = 0;
       }
       if (gpkt->payload_size) {
               rte_memcpy(&gpkt->payload, datastart, gpkt->payload_size);
       }
       size_t bsz = sizeof(gpu_packet_t) + gpkt->payload_size;
       if (bsz % GPU_PKT_ALIGN != 0) {
               bsz += GPU_PKT_ALIGN - (bsz % GPU_PKT_ALIGN);
       }

#ifdef END_TO_END_LATENCY
        static long freq = 0;
        struct timespec timestamp;
        if (freq % 1000000 == 0) {
            freq = 0;
            clock_gettime(CLOCK_MONOTONIC, &timestamp);
            // measure time by inserting magic number and current time in packet
            gpkt->src_addr = LATENCY_MAGIC;
            gpkt->dst_addr = (uint32_t) timestamp.tv_sec;
            gpkt->sent_seq = (uint32_t) timestamp.tv_nsec;
            gpkt->recv_ack = (uint32_t) (timestamp.tv_nsec >> 32);
        }
        freq++;
#endif

       return bsz;
}

static size_t unload_packet(uint8_t *buffer, struct rte_mbuf *pkt) {
    // todo: update packet size
    gpu_packet_t *gpkt = (gpu_packet_t *) buffer;
    //RTE_LOG(INFO, APP, "unload %u\n", gpkt->payload_size);
    uint8_t *datastart = NULL;//, *dataend;
    struct ipv4_hdr* ipv4 = onvm_pkt_ipv4_hdr(pkt);
    if (!ipv4) {
        return sizeof(gpu_packet_t);
    }

    if(((last_plan>>7) & 1) == 1)
        ipv4->src_addr = gpkt->src_addr;

    if(((last_plan>>6) & 1) == 1)
        ipv4->dst_addr = gpkt->dst_addr;

    //ipv4->next_proto_id = gpkt->proto_id;
    if (ipv4->next_proto_id == IPPROTO_TCP) {
        struct tcp_hdr *tcp = onvm_pkt_tcp_hdr(pkt);
        tcp->tcp_flags = gpkt->tcp_flags;
        // tcp->src_port = rte_be_to_cpu_16(gpkt->src_port);
        if(((last_plan>>5) & 1) == 1)
            tcp->src_port = rte_cpu_to_be_16(gpkt->src_port);

        if(((last_plan>>4) & 1) == 1)  
            tcp->dst_port = rte_cpu_to_be_16(gpkt->dst_port);

        tcp->sent_seq = rte_cpu_to_be_32(gpkt->sent_seq);
        tcp->recv_ack = rte_cpu_to_be_32(gpkt->recv_ack);
        datastart = (uint8_t *)tcp + (tcp->data_off >> 4 << 2);
        //dataend = datastart + gpkt->payload_size;
        //ipv4->total_length = rte_be_to_cpu_16(dataend - (uint8_t *)ipv4);
    } else if (ipv4->next_proto_id == IPPROTO_UDP) {
        struct udp_hdr *udp = onvm_pkt_udp_hdr(pkt);
        if(((last_plan>>5) & 1) == 1)
            udp->src_port = rte_cpu_to_be_16(udp->src_port);

        if(((last_plan>>4) & 1) == 1)  
            udp->dst_port = rte_cpu_to_be_16(udp->dst_port);

        datastart = (uint8_t *)udp + 8;
        udp->dgram_len = rte_cpu_to_be_16(gpkt->payload_size + 8);
    }
    if (datastart && gpkt->payload_size) {
        rte_memcpy(datastart, gpkt->payload, gpkt->payload_size);
    }
    size_t bsz = sizeof(gpu_packet_t) + gpkt->payload_size;
    if (bsz % GPU_PKT_ALIGN != 0) {
        bsz += GPU_PKT_ALIGN - (bsz % GPU_PKT_ALIGN);
    }
#ifdef END_TO_END_LATENCY
    if (unlikely(gpkt->src_addr == LATENCY_MAGIC)) {
        struct timespec prev;
        prev.tv_sec = gpkt->dst_addr;
        prev.tv_nsec = (uint64_t) gpkt->sent_seq | ((uint64_t) gpkt->recv_ack << 32);
        double timediff_usec = time_diff(prev);
        printf("End to end latency: %.3f ms\n", timediff_usec / 1e3);
    }
#endif
    return bsz;
}

static int
rx_thread_main(void *arg) {
    unsigned i, j, rx_count, rx_len;
    //64
    struct rte_mbuf *pkts[PACKET_READ_SIZE];
    //这里的rx代表rx队列
    struct thread_info *rx = (struct thread_info*)arg;
    unsigned int core_id = rte_lcore_id();
    int thread_id = rx->queue_id;

    unsigned rx_batch_id;

    RTE_LOG(INFO, APP, "Core %d: Running RX thread for RX queue %d\n", core_id, rx->queue_id);

    rx_batch_id = 0;
    unsigned batch_head = 0;
    unsigned batch_cnt = 0;
    for (;;) {
		/* Read ports */
		for (i = 0; i < ports->num_ports; i++) {
            rx_len = 0;
            //读取端口号数据
			rx_count = rte_eth_rx_burst(ports->id[i], rx->queue_id, pkts, PACKET_READ_SIZE);
            if (rx_count > 0) {
                rx_len += (*pkts[0]).pkt_len * rx_count;
            }

            for (j = 0; j < rx_count; j++) {
                struct onvm_pkt_meta *meta = onvm_get_pkt_meta(pkts[j]);
                meta->action = ONVM_NF_ACTION_TONF;
                //得到每一个接受到的数据包的大小，这个函数最终返回的数据包大小是应用层数据大小加上gpu头部信息
                unsigned pkt_sz = size_packet(pkts[j]);
                
                if (batch_head + pkt_sz > RX_BUF_SIZE) {
                    //rx_batch_id有何用？？？
                    //rx线程是按批来处理数据包的，一次最多处理四个，rx_batch可以理解为批处理数组？
                    unsigned next_id = (rx_batch_id + 1) % RX_NUM_BATCHES;
                    if (rx_batch[next_id].full[thread_id]) break;
                    rx_batch[rx_batch_id].pkt_cnt[thread_id] = batch_cnt;
                    batch_cnt = 0;
                    batch_head = 0;
#ifndef DROP_RX_PKTS
                    rx_batch[rx_batch_id].full[thread_id] = 1;
#endif
                    rx_batch_id = next_id;
                }
                //在指定批的指定处理线程上添加所得到的数据
                //首先是dpdk层数据得到
                rx_batch[rx_batch_id].pkt_ptr[thread_id][batch_cnt++] = pkts[j];
                //接着是用户层数据得到，并且转换成cuda能用的形式，pos表示用户层能用的数据
                uint8_t *pos = rx_batch[rx_batch_id].buf[thread_id] + batch_head;
                //这个buf_head本来就是指的gpu层的信息
                onvm_pkt_gpu_ptr(pkts[j]) = rx_batch[rx_batch_id].buf_head + (pos - (uint8_t *)&rx_batch[rx_batch_id].buf);
                
                //将已经转换成cuda模式的pkts数据包传递给用户空间，最终得到下一个数据包的头部位置 
                batch_head += load_packet(pos, pkts[j]);

                // Use pstack to process TCP/IP
#ifdef ENABLE_PSTACK
                void* res = pstack_process((char *)onvm_pkt_ipv4_hdr(pkts[j]), pkts[j]->data_len - sizeof(struct ether_hdr), thread_id);
#endif
            }
#ifndef DROP_RX_PKTS
            if (unlikely(j < rx_count)) {
                onvm_pkt_drop_batch(&pkts[j], rx_count - j);
            }
#else
            onvm_pkt_drop_batch(pkts, rx_count);
#endif
            rte_atomic64_add((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx[0], rx_count);
            rte_atomic64_add((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx_len[0], rx_len);
		}
	}

	return 0;
}


static void cu_memcpy_cb(CUstream stream, CUresult result, void *data) {
    UNUSED(stream);
    checkCudaErrors(result);
    volatile int *sync = (volatile int *)data;
    *sync = 1;
}

//将rx数据给服务链第一个服务的rss数组，并且将其传递给cuda内存
static int
rx_gpu_thread_main(void *arg) {
    UNUSED(arg);
    checkCudaErrors( cuInit(0) );
    checkCudaErrors( cuCtxSetCurrent(context) );

    CUstream stream;
    checkCudaErrors( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );

    int batch_id = 0, tonf_id = 0;
    unsigned i;
    unsigned rx_count, rx_count_total, rx_len_total;
    UNUSED(rx_count);
    struct rte_ring **rx_q_new = NULL;
    if (default_chain->sc[1].action == ONVM_NF_ACTION_OUT)
    {
	// rx_q_new = ports->tx_q_new[default_chain->sc[1].destination];
        rx_q_new = ports->tx_qs[default_chain->sc[1].destination];
	printf("get port tx\n");
    }
    else if (default_chain->sc[1].action != ONVM_NF_ACTION_TONF)
        rte_exit(EXIT_FAILURE, "Failed to find first nf");
    uint16_t first_service_id = default_chain->sc[1].destination;

    //当前得到的局部rx批
    rx_batch_t *batch;

    for (;;) {
        rx_count_total = 0, rx_len_total = 0;
        //从0开始处理rx批，每次都会
        batch = &rx_batch[tonf_id];
        if (batch->gpu_sync) {
            if (gpu_pkts_head + sizeof(batch->buf) > gpu_pkts_buf + GPU_BUF_SIZE * GPU_MAX_PKT_LEN) {
                gpu_pkts_head = gpu_pkts_buf;
            }
            //gpu_pkts_head更像是指向头部的指针信息
            batch->buf_head = gpu_pkts_head;
            gpu_pkts_head += sizeof(batch->buf);
            batch->gpu_sync = 0;
            tonf_id = (tonf_id + 1) % RX_NUM_BATCHES;

            if (unlikely(rx_q_new == NULL)) {
                if (nf_per_service_count[first_service_id] > 0) {
                    // rx_q_new = clients[services[first_service_id][0]].rx_q_new;
                    rx_q_new = clients[services[first_service_id][0]].rx_qs;
                }
            }

            //遍历当前批中所有线程缓冲区的数据
            for (i = 0; i < RX_NUM_THREADS; i++) {
                rx_count = 0;
                if (rx_q_new != NULL) {
#ifdef RING_QUEUING_LATENCY
                    static int freq = 0;
                    if (likely(batch->pkt_cnt[i] > 0)) {
                        for (unsigned n = 0; n < batch->pkt_cnt[i]; n++)
                                batch->pkt_ptr[i][n]->seqn = 0;
                        if (freq % 50000 == 0) {
                            struct timespec cur;
                            clock_gettime(CLOCK_MONOTONIC, &cur);
                            batch->pkt_ptr[i][0]->seqn = LATENCY_MAGIC;
                            
                            batch->pkt_ptr[i][0]->tv_nsec = cur.tv_nsec;
                            batch->pkt_ptr[i][0]->tv_sec = cur.tv_sec;
                            freq = 0;
                        }
                        freq++;
                    }
#endif
                    // while (rx_count == 0)
                    //     rx_count = rte_ring_enqueue_bulk(rx_q_new, (void **)batch->pkt_ptr[i], batch->pkt_cnt[i], NULL);

                    //这一部操作就相当于把dpdk层获得的数据复制给了对应的nf层的缓冲区中rss helper
                    while (rx_count == 0)
                         rx_count = rte_ring_enqueue_bulk(rx_q_new[i % ONVM_NUM_NF_QUEUES], (void **)batch->pkt_ptr[i], batch->pkt_cnt[i], NULL);
                }
                if (rx_count < batch->pkt_cnt[i]) {
                    // it takes some time so performance is worse if all dropped
                    onvm_pkt_drop_batch(batch->pkt_ptr[i] + rx_count, batch->pkt_cnt[i] - rx_count);
                }
                batch->full[i] = 0;
                rx_count_total += batch->pkt_cnt[i];
                rx_len_total += batch->pkt_cnt[i] == 0 ? 0 : batch->pkt_cnt[i] * batch->pkt_ptr[i][0]->pkt_len;
#ifdef MEASURE_RX_LATENCY
                double timediff_usec = time_diff(batch->batch_start_time[i]);
                printf("RX latency: %.3f ms\n", timediff_usec / 1e3);

                clock_gettime(CLOCK_MONOTONIC, &(batch->batch_start_time[i]));
#endif
            }
        }

        rte_atomic64_add((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx_gpucopy, rx_count_total);
        rte_atomic64_add((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx_len_gpucopy, rx_len_total);

        if (batch_id + 1 == tonf_id || batch_id + 1 - RX_NUM_BATCHES == tonf_id) continue;
        batch = &rx_batch[batch_id];
        int full = 1;
        for (i = 0; i < RX_NUM_THREADS; i++) {
            full = (full && batch->full[i]);
        }
        if (!full) continue;

        checkCudaErrors( cuMemcpyHtoDAsync(batch->buf_head, (void *)batch->buf, sizeof(batch->buf), stream) );
        checkCudaErrors( cuStreamAddCallback(stream, cu_memcpy_cb, (void *)(uintptr_t)&batch->gpu_sync, 0) );

        batch_id = (batch_id + 1) % RX_NUM_BATCHES;
    }

	return 0;
}

//从gpu中获取数据，并且进行转发
static int
tx_thread_main(void *arg) {
	struct thread_info *tx = (struct thread_info*)arg;
	unsigned int core_id = rte_lcore_id();
    int thread_id = tx->queue_id;

    //tx线程也是批处理的
    tx_batch_t *gpu_batch = &tx_batch[thread_id];

	unsigned i, sent;
    unsigned gpu_packet = 0;
    unsigned remain = 0;

    //dpdk数据
    struct rte_mbuf **gpu_batching;

    //用户层数据
    uint8_t *batch_buffer;
    CUdeviceptr batch_buffer_base;

    volatile int sync = 1;

    unsigned batch_id = 0;

	checkCudaErrors( cuInit(0) );
    //这些个线程都是共享一个由manager创建的上下文的
	checkCudaErrors( cuCtxSetCurrent(context) );

	gpu_batching = rte_calloc("tx gpu batch", TX_GPU_BATCH_SIZE, sizeof(struct rte_mbuf *), 0);
	checkCudaErrors( cuMemAllocHost((void **)&batch_buffer, TX_GPU_BUF_SIZE) );

    gpu_batch->pkt_ptr = gpu_batching;
    gpu_batch->buf = batch_buffer;

	CUstream stream;
	checkCudaErrors( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );

	RTE_LOG(INFO, APP, "Core %d: Running TX thread for port %d\n", core_id, tx->port_id);

    //表示是否要遍历nf
    uint8_t iterate_queue = ONVM_NUM_NF_QUEUES > ONVM_NUM_TX_THREADS_PER_PORT;
    //表示是否要遍历端口信息
    uint8_t multi_consumers = ONVM_NUM_NF_QUEUES < ONVM_NUM_TX_THREADS_PER_PORT;
    current_tx_qid = iterate_queue ? thread_id : (thread_id % ONVM_NUM_NF_QUEUES);

    for (;;) {
        if (sync) {
            if (gpu_packet > 0) {
                //这个unloadable是用来干什么的
                unsigned unloadable = 0;
                while (unloadable < gpu_packet) {
                    unsigned pkt_off = onvm_pkt_gpu_ptr(gpu_batching[unloadable]) - batch_buffer_base;
                    //偏移量越界
                    if (pkt_off > TX_GPU_BUF_SIZE - sizeof(gpu_packet_t)) break;
                    //负载越界
                    if (((gpu_packet_t *)(batch_buffer + pkt_off))->payload_size > 
                            TX_GPU_BUF_SIZE - sizeof(gpu_packet_t) - pkt_off) break;
                    //unload_packet(batch_buffer + pkt_off, gpu_batching[unloaded]);
                    //unloaded++;
                    unloadable++;
                }
                ports->tx_stats.gpu_batch_cnt[tx->port_id]++;
                ports->tx_stats.gpu_batch_pkt[tx->port_id] += unloadable;
                /*
                unsigned queued = rte_ring_enqueue_burst(gpu_q, (void **)gpu_batching, unloaded, NULL);
                if (unlikely(queued < unloaded)) {
                    onvm_pkt_drop_batch(gpu_batching + queued, unloaded - queued);
                    ports->tx_stats.tx_drop[tx->port_id] += unloaded - queued;
                }
                */
                remain = gpu_packet - unloadable;
                gpu_packet = 0;
                gpu_batch->pkt_cnt = unloadable;
                gpu_batch->buf_base = batch_buffer_base;
                rte_wmb();
                for (i = 0; i < ONVM_NUM_TX_THREADS_PER_PORT; i++) {
                    gpu_batch->ready[i] = 1;
                }
            }
            
            //ready表示准备转发,下述循环表示是否每个线程都准备进行批处理了
            //只要有一个可以进入ready状态，就对那一个进行批处理
            int ready = 0;
            for (i = 0; i < ONVM_NUM_TX_THREADS_PER_PORT; i++) {
                ready = (ready || gpu_batch->ready[i]);
            }

            //如果上述过程中，batch的每个线都都为0
            if (ready == 0) {
                memmove(gpu_batching, gpu_batching + gpu_batch->pkt_cnt, remain * sizeof(struct rte_mbuf *));
                gpu_packet = remain;
                if (gpu_packet < TX_GPU_BATCH_SIZE / 2) {
                    // gpu_packet += rte_ring_dequeue_burst(
                    //     ports->tx_q_new[tx->port_id], (void **)(gpu_batching + remain), TX_GPU_BATCH_SIZE - remain, NULL);
                    if (multi_consumers) {
                        gpu_packet += rte_ring_mc_dequeue_burst(
                            ports->tx_qs[tx->port_id][current_tx_qid], (void **)(gpu_batching + remain), TX_GPU_BATCH_SIZE - remain, NULL);
                    } else {
                        gpu_packet += rte_ring_sc_dequeue_burst(
                            ports->tx_qs[tx->port_id][current_tx_qid], (void **)(gpu_batching + remain), TX_GPU_BATCH_SIZE - remain, NULL);
                    }
                }
                if (likely(gpu_packet > 0)) {
                    batch_buffer_base = onvm_pkt_gpu_ptr(gpu_batching[0]);
                    checkCudaErrors( cuMemcpyDtoHAsync(batch_buffer, batch_buffer_base, TX_GPU_BUF_SIZE, stream) );
                    sync = 0;
                    checkCudaErrors( cuStreamAddCallback(stream, cu_memcpy_cb, (void *)(uintptr_t)&sync, 0) );
                }
            }
        }

        if (iterate_queue) {
            current_tx_qid = current_tx_qid + ONVM_NUM_TX_THREADS_PER_PORT;
            if (current_tx_qid >= ONVM_NUM_NF_QUEUES) {
                current_tx_qid = thread_id;
            }
        }

        for (i = 0; i < ONVM_NUM_TX_THREADS_PER_PORT; i++) {
            if (tx_batch[batch_id].ready[thread_id]) {
                break;
            }
            batch_id = (batch_id + 1) % ONVM_NUM_TX_THREADS_PER_PORT;
        }
        if (i == ONVM_NUM_TX_THREADS_PER_PORT) {
            batch_id = (batch_id + 1) % ONVM_NUM_TX_THREADS_PER_PORT;
            continue;
        }

        unsigned range_id = (thread_id + batch_id) % ONVM_NUM_TX_THREADS_PER_PORT;
        unsigned step = (tx_batch[batch_id].pkt_cnt + ONVM_NUM_TX_THREADS_PER_PORT - 1) / ONVM_NUM_TX_THREADS_PER_PORT;
        unsigned cnt = (step * (range_id + 1) <= tx_batch[batch_id].pkt_cnt ? step : (
                        step * range_id <= tx_batch[batch_id].pkt_cnt ? tx_batch[batch_id].pkt_cnt - step * range_id : 0));

        sent = 0;
        for (i = 0; i < cnt; i++) {
            unsigned pkt_off = onvm_pkt_gpu_ptr(tx_batch[batch_id].pkt_ptr[step * range_id + i]) - tx_batch[batch_id].buf_base;
            unload_packet(tx_batch[batch_id].buf + pkt_off, tx_batch[batch_id].pkt_ptr[step * range_id + i]);
            if (i > 0 && i % 64 == 0)
                sent += rte_eth_tx_burst(tx->port_id, tx->queue_id, tx_batch[batch_id].pkt_ptr + step * range_id + sent, i - sent);
        }

        unsigned pkt_size_sample = cnt > 0 ? tx_batch[batch_id].pkt_ptr[step * range_id]->pkt_len : 0;

        while (sent < cnt) {
            sent += rte_eth_tx_burst(tx->port_id, tx->queue_id, tx_batch[batch_id].pkt_ptr + step * range_id + sent, cnt - sent);
        }
        tx_batch[batch_id].ready[thread_id] = 0;

#ifdef MEASURE_TX_LATENCY
        printf("Tx latency: %.3f ms\n", time_diff(tx_batch[batch_id].batch_start_time[thread_id]));

        clock_gettime(CLOCK_MONOTONIC, &(tx_batch[batch_id].batch_start_time[thread_id]));
#endif
        // ports->tx_stats.tx[tx->port_id] += cnt;
        rte_atomic64_add((rte_atomic64_t *)(uintptr_t)&ports->tx_stats.tx[tx->port_id], cnt);
        rte_atomic64_add((rte_atomic64_t *)(uintptr_t)&ports->tx_stats.tx_len[tx->port_id], cnt * pkt_size_sample);
	}

	return 0;
}

/*******************************Main function*********************************/

int
main(int argc, char *argv[]) {
	//signal(SIGSEGV, segv_handler);
	unsigned cur_lcore, rx_lcores, tx_lcores;
	unsigned i, j;

	/* initialise the system */

	/* Reserve ID 0 for internal manager things */
	next_instance_id = 1;
	if (init(argc, argv) < 0)
		return -1;
	RTE_LOG(INFO, APP, "Finished Process Init.\n");

	/* clear statistics */
	onvm_stats_clear_all_clients();

	/* Reserve n cores for: 1 Scheduler + Stats, 1 Manager, 1 GPU Thread, and ONVM_NUM_RX_THREADS for Rx, 1 per port for Tx, to be adjusted */
	cur_lcore = rte_lcore_id();
	rx_lcores = RX_NUM_THREADS;
	tx_lcores = ports->num_ports * ONVM_NUM_TX_THREADS_PER_PORT;

	RTE_LOG(INFO, APP, "%d cores available in total\n", rte_lcore_count());
	RTE_LOG(INFO, APP, "%d cores available for handling RX queues\n", rx_lcores);
	RTE_LOG(INFO, APP, "%d cores available for handling TX queues\n", tx_lcores);
	RTE_LOG(INFO, APP, "%d cores available for Manager\n", 1);
	RTE_LOG(INFO, APP, "%d cores available for Scheduler + States\n", 1);

	if (rx_lcores + tx_lcores + 3 > rte_lcore_count()) {
		rte_exit(EXIT_FAILURE, "%d cores needed, but %d cores specified\n", rx_lcores+tx_lcores+3, rte_lcore_count());
	}

	// We start the system with 0 NFs active
	num_clients = 0;

	/* Launch Manager thread as the GPU proxy */
    //启动状态机和调度器线程
	cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
	if (rte_eal_remote_launch(manager_thread_main, NULL, cur_lcore) == -EBUSY) {
		RTE_LOG(ERR, APP, "Core %d is already busy, can't use for Manager\n", cur_lcore);
		return -1;
	}
#ifndef DISABLE_TX
    //启动端口号*tx个数各线程
	/* Assign each port with a TX thread */
	for (i = 0; i < ports->num_ports; i++) {
        for (j = 0; j < ONVM_NUM_TX_THREADS_PER_PORT; j++) {
            struct thread_info *tx = calloc(1, sizeof(struct thread_info));
            tx->port_id = ports->id[i]; /* Actually this is the port id */
            tx->queue_id = j;
            tx->port_tx_buf = calloc(PACKET_WRITE_SIZE, sizeof(struct rte_mbuf *));

            cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
            if (rte_eal_remote_launch(tx_thread_main, (void*)tx,  cur_lcore) == -EBUSY) {
                RTE_LOG(ERR, APP, "Core %d is already busy, can't use for port %d TX\n", cur_lcore, tx->queue_id);
                return -1;
            }
        }
	}
#endif
    /* init rx bufs */
    for (i = 0; i < RX_NUM_BATCHES; i++) {
        rx_batch[i].gpu_sync = 0;
        rx_batch[i].buf_head = gpu_pkts_head;
        gpu_pkts_head += sizeof(rx_batch[i].buf);
        for (j = 0; j < RX_NUM_THREADS; j++) {
            rx_batch[i].full[j] = 0;
        }
    }

	/* Launch RX thread main function for each RX queue on cores */
    //启动rx个数各线程
	for (i = 0; i < rx_lcores; i++) {
		struct thread_info *rx = calloc(1, sizeof(struct thread_info));
		rx->queue_id = i;
		rx->port_tx_buf = NULL;

		cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
		if (rte_eal_remote_launch(rx_thread_main, (void *)rx, cur_lcore) == -EBUSY) {
			RTE_LOG(ERR, APP, "Core %d is already busy, can't use for RX queue id %d\n", cur_lcore, rx->queue_id);
			return -1;
		}
	}
#ifndef DISABLE_GPU_RX
    //gpu线程
    cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
    if (rte_eal_remote_launch(rx_gpu_thread_main, NULL, cur_lcore) == -EBUSY) {
        RTE_LOG(ERR, APP, "Core %d is already busy, can't use for RX GPU\n", cur_lcore);
        return -1;
    }
#endif
	/* Master thread handles statistics and resource allocation */
    //当前线程用来充当调度器
	scheduler_thread_main(NULL);
	return 0;
}
