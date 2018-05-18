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

extern struct onvm_service_chain *default_chain;
extern struct rx_perf rx_stats[ONVM_NUM_RX_THREADS]; 
extern struct port_info *ports;
extern CUdeviceptr gpu_pkts_buf;
extern CUdeviceptr gpu_pkts_head;
extern rte_spinlock_t gpu_pkts_lock;
extern CUcontext context;

/*******************************Worker threads********************************/

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
               datastart = (uint8_t *)tcp + (tcp->data_off >> 2);
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
               memcpy(&gpkt->payload, datastart, gpkt->payload_size);
       }
       size_t bsz = sizeof(gpu_packet_t) + gpkt->payload_size;
       if (bsz % 16 != 0) {
               bsz += 16 - (bsz % 16);
       }
       return bsz;
}

/*
 * Function to receive packets from the NIC
 * and distribute them to the default service
 */
static int
rx_thread_main(void *arg) {
	unsigned i, j, rx_count;
	struct rte_mbuf *pkts[PACKET_READ_SIZE];
	struct rte_mbuf **gpu_batching;
	uint8_t *batch_buffer;
	struct thread_info *rx = (struct thread_info*)arg;
	unsigned int core_id = rte_lcore_id();
	uint64_t starve_rx_counter = 0;	

	size_t num_gpu_batch = 0;
	size_t queued;

	checkCudaErrors( cuInit(0) );
	checkCudaErrors( cuCtxSetCurrent(context) );

	gpu_batching = rte_calloc("rx gpu batch", RX_GPU_BATCH_SIZE, sizeof(struct rte_mbuf *), 0);
	checkCudaErrors( cuMemAllocHost((void **)&batch_buffer, RX_GPU_BATCH_SIZE * GPU_PKT_LEN) );

	RTE_LOG(INFO, APP, "Core %d: Running RX thread for RX queue %d\n", core_id, rx->queue_id);
	
	struct rte_ring *rx_q_new = NULL;
	if (default_chain->sc[1].action == ONVM_NF_ACTION_OUT)
			rx_q_new = ports->tx_q_new[default_chain->sc[1].destination];
	else if (default_chain->sc[1].action != ONVM_NF_ACTION_TONF)
			rte_exit(EXIT_FAILURE, "Failed to find first nf");
	uint16_t first_service_id = default_chain->sc[1].destination;

	struct rte_ring *gpu_q = rte_ring_lookup(RX_GPU_QUEUE);
	if (!gpu_q)
		rte_exit(EXIT_FAILURE, "RX_GPU_Q not found");
	CUstream stream;
	checkCudaErrors( cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) );
	CUevent gpu_state;
	checkCudaErrors( cuEventCreate(&gpu_state, CU_EVENT_DEFAULT) );


	for (;;) {
		if (CUDA_SUCCESS == cuEventQuery(gpu_state)) { // available
			if (likely(num_gpu_batch > 0)) {
				if (unlikely(rx_q_new == NULL)) {
					if (nf_per_service_count[first_service_id] > 0) {
						rx_q_new = clients[services[first_service_id][0]].rx_q_new;
					}
				}
				if (likely(rx_q_new != NULL)) {
					queued = rte_ring_enqueue_burst(rx_q_new, (void **)gpu_batching, num_gpu_batch, NULL);
					if (unlikely(queued < num_gpu_batch)) {
						onvm_pkt_drop_batch(gpu_batching + queued, num_gpu_batch - queued);
					}
				} else {
					onvm_pkt_drop_batch(gpu_batching, num_gpu_batch);
				}
			}
			num_gpu_batch = rte_ring_dequeue_bulk(gpu_q, (void **)gpu_batching, RX_GPU_BATCH_SIZE, NULL);
			if (num_gpu_batch > 0) {
				size_t buf_sz = 0, cur_sz;
				for (i = 0; i < num_gpu_batch; i++) {
					cur_sz = load_packet(batch_buffer + buf_sz, gpu_batching[i]);
					onvm_pkt_gpu_ptr(gpu_batching[i]) = cur_sz;
					buf_sz += cur_sz;
				}
				CUdeviceptr head;
				rte_spinlock_lock(&gpu_pkts_lock);
				head = gpu_pkts_head;
				if (head + buf_sz > gpu_pkts_buf + GPU_BUF_SIZE * GPU_PKT_LEN) {
					head = gpu_pkts_buf;
				}
				gpu_pkts_head = head + buf_sz;
				rte_spinlock_unlock(&gpu_pkts_lock);
				//checkCudaErrors( cuMemcpyHtoDAsync(head, batch_buffer, buf_sz, stream) );
				checkCudaErrors( cuEventRecord(gpu_state, stream) );
				for (i = 0; i < num_gpu_batch; i++) {
					onvm_pkt_gpu_ptr(gpu_batching[i]) += head;
				}
			}
		}

		/* Read ports */
		for (i = 0; i < ports->num_ports; i++) {
			rx_count = rte_eth_rx_burst(ports->id[i], rx->queue_id, pkts, PACKET_READ_SIZE);
			ports->rx_stats.rx[ports->id[i]] += rx_count;

			/* Now process the NIC packets read */
			if (likely(rx_count > 0)) {
				starve_rx_counter = 0;
				for (j = 0; j < rx_count; j++) {
					// workaround: mark not to drop the packet
					struct onvm_pkt_meta *meta = onvm_get_pkt_meta(pkts[j]);
					meta->action = ONVM_NF_ACTION_TONF;
				}
				queued = rte_ring_enqueue_burst(gpu_q, (void **)pkts, rx_count, NULL);
				if (unlikely(queued < rx_count)) {
					onvm_pkt_drop_batch(pkts + queued, rx_count - queued);
				}
			} else {
				starve_rx_counter++;
				if (starve_rx_counter == STARVE_THRESHOLD) {
					RTE_LOG(INFO, APP, "Rx starving\n");
				}
			}
		}
	}

	return 0;
}

static int
tx_thread_main(void *arg) {
	struct thread_info *tx = (struct thread_info*)arg;
	unsigned int core_id = rte_lcore_id();

	unsigned tx_count;
	unsigned sent;

	RTE_LOG(INFO, APP, "Core %d: Running TX thread for port %d\n", core_id, tx->port_id);

	for (;;) {
		tx_count = rte_ring_dequeue_burst(
			ports->tx_q_new[tx->port_id], (void **)tx->port_tx_buf, PACKET_WRITE_SIZE, NULL);
		if (likely(tx_count > 0)) {
			sent = rte_eth_tx_burst(tx->port_id, 0, tx->port_tx_buf, tx_count);
			onvm_pkt_drop_batch(tx->port_tx_buf + sent, tx_count - sent);
			ports->tx_stats.tx[tx->port_id] += sent;
			ports->tx_stats.tx_drop[tx->port_id] += sent;
		}
	}

	return 0;
}

/*******************************Main function*********************************/

int
main(int argc, char *argv[]) {
	//signal(SIGSEGV, segv_handler);
	unsigned cur_lcore, rx_lcores, tx_lcores;
	unsigned i;

	/* initialise the system */

	/* Reserve ID 0 for internal manager things */
	next_instance_id = 1;
	if (init(argc, argv) < 0)
		return -1;
	RTE_LOG(INFO, APP, "Finished Process Init.\n");

	/* clear statistics */
	onvm_stats_clear_all_clients();

	/* Reserve n cores for: 1 Scheduler + Stats, 1 Manager, and ONVM_NUM_RX_THREADS for Rx, 1 per port for Tx, to be adjusted */
	cur_lcore = rte_lcore_id();
	rx_lcores = ONVM_NUM_RX_THREADS;
	tx_lcores = ports->num_ports;

	RTE_LOG(INFO, APP, "%d cores available in total\n", rte_lcore_count());
	RTE_LOG(INFO, APP, "%d cores available for handling RX queues\n", rx_lcores);
	RTE_LOG(INFO, APP, "%d cores available for handling TX queues\n", tx_lcores);
	RTE_LOG(INFO, APP, "%d cores available for Manager\n", 1);
	RTE_LOG(INFO, APP, "%d cores available for Scheduler + States\n", 1);

	if (rx_lcores + tx_lcores + 2 != rte_lcore_count()) {
		rte_exit(EXIT_FAILURE, "%d cores needed, but %d cores specified\n", rx_lcores+tx_lcores+2, rte_lcore_count());
	}

	// We start the system with 0 NFs active
	num_clients = 0;

	/* Launch Manager thread as the GPU proxy */
	cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
	if (rte_eal_remote_launch(manager_thread_main, NULL, cur_lcore) == -EBUSY) {
		RTE_LOG(ERR, APP, "Core %d is already busy, can't use for Manager\n", cur_lcore);
		return -1;
	}

	/* Assign each port with a TX thread */
	for (i = 0; i < tx_lcores; i++) {
		struct thread_info *tx = calloc(1, sizeof(struct thread_info));
		tx->port_id = ports->id[i]; /* Actually this is the port id */
		tx->port_tx_buf = calloc(PACKET_WRITE_SIZE, sizeof(struct rte_mbuf *));

		cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
		if (rte_eal_remote_launch(tx_thread_main, (void*)tx,  cur_lcore) == -EBUSY) {
			RTE_LOG(ERR, APP, "Core %d is already busy, can't use for port %d TX\n", cur_lcore, tx->queue_id);
			return -1;
		}
	}

	/* Launch RX thread main function for each RX queue on cores */
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

	/* Master thread handles statistics and resource allocation */
	scheduler_thread_main(NULL);
	return 0;
}
