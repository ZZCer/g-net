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
                                 onvm_pkt.c

            This file contains all functions related to receiving or
            transmitting packets.

******************************************************************************/


#include "onvm_mgr.h"
#include "onvm_pkt.h"
#include "onvm_nf.h"
#include "onvm_common.h"
#include "scheduler.h"
#include "fifo.h"


/**********************Internal Functions Prototypes**************************/


/*
 * Function to send packets to one port after processing them.
 *
 * Input : a pointer to the tx queue
 *
 */
static void
onvm_pkt_flush_port_queue(struct thread_info *tx, uint16_t port);


/*
 * Function to send packets to one NF after processing them.
 *
 * Input : a pointer to the tx queue
 *
 */
static void
onvm_pkt_flush_nf_queue(struct thread_info *thread, uint16_t client);


/*
 * Function to enqueue a packet on one port's queue.
 *
 * Inputs : a pointer to the tx queue responsible
 *          the number of the port
 *          a pointer to the packet
 *
 */
inline static void
onvm_pkt_enqueue_port(struct thread_info *tx, uint16_t port, struct rte_mbuf *buf);


/*
 * Function to enqueue a packet on one NF's queue.
 *
 * Inputs : a pointer to the tx queue responsible
 *          the number of the port
 *          a pointer to the packet
 *
 */
inline static void
onvm_pkt_enqueue_nf(struct thread_info *thread, uint16_t dst_service_id, struct rte_mbuf *pkt);


/*
 * Function to process a single packet.
 *
 * Inputs : a pointer to the tx queue responsible
 *          a pointer to the packet
 *          a pointer to the NF involved
 *
 */
inline static void
onvm_pkt_process_next_action(struct thread_info *tx, struct rte_mbuf *pkt, struct client *cl);


/*
 * Helper function to drop a packet.
 *
 * Input : a pointer to the packet
 *
 * Ouput : an error code
 *
 */
static int
onvm_pkt_drop(struct rte_mbuf *pkt);


/**********************************Interfaces*********************************/

#if defined(RX_SPEED_TEST_2)
extern struct rx_perf rx_stats[ONVM_NUM_RX_THREADS]; 
#endif

void
onvm_pkt_process_rx_batch(struct thread_info *rx, struct rte_mbuf *pkts[], uint16_t rx_count) {
	uint16_t i;
	struct onvm_pkt_meta *meta;
	struct onvm_flow_entry *flow_entry;
	struct onvm_service_chain *sc;
	int ret;

	if (rx == NULL || pkts == NULL)
		return;

	for (i = 0; i < rx_count; i++) {
		meta = (struct onvm_pkt_meta*) &(((struct rte_mbuf*)pkts[i])->udata64);
		meta->src = 0;
		meta->chain_index = 0;

		ret = onvm_flow_dir_get_pkt(pkts[i], &flow_entry);
		if (ret >= 0) {
			rte_exit(EXIT_FAILURE, "Not support SDN flow table yet\n");
			sc = flow_entry->sc;
			meta->action = onvm_sc_next_action(sc, pkts[i]);
			meta->destination = onvm_sc_next_destination(sc, pkts[i]);
		} else {
			meta->action = onvm_sc_next_action(default_chain, pkts[i]);
			meta->destination = onvm_sc_next_destination(default_chain, pkts[i]);
		}
		/* PERF: this might hurt performance since it will cause cache
		 * invalidations. Ideally the data modified by the NF manager
		 * would be a different line than that modified/read by NFs.
		 * That may not be possible.
		 */

		(meta->chain_index) ++;

#if defined(BQUEUE_SWITCH)
		uint16_t dst_instance_id = onvm_nf_service_to_nf_map(meta->destination, pkts[i]);
		if (dst_instance_id == 0) {
			onvm_pkt_drop(pkts[i]);
			return;
		}

		struct client *cl = &clients[dst_instance_id];
		if (unlikely(!onvm_nf_is_valid(cl))) {
			onvm_pkt_drop(pkts[i]);
			return;
		}

		if (unlikely(cl->stats.reset == 1 && rx->queue_id == 0)) {
			cl->stats.reset = 0;
			cl->stats.rx = 0;
			cl->stats.rx_datalen = 0;
			cl->stats.rx_drop = 0;
			clock_gettime(CLOCK_MONOTONIC, &(cl->stats.start));
		}

#if defined(MATCH_RX_THREAD)
		if (unlikely(ONVM_NUM_RX_THREADS != cl->gpu_info->thread_num)) {
			rte_exit(EXIT_FAILURE, "Thread number not match %d != %d\n",
					ONVM_NUM_RX_THREADS, cl->gpu_info->thread_num);
		}
#endif

#if defined(RX_SPEED_TEST_2)
		rte_pktmbuf_free(pkts[i]);
		continue;
#endif

		/* Enqueue to the corresponding queue */
		if (SUCCESS == bq_enqueue(cl->rx_bq[rx->queue_id], pkts[i])) {
			if (rx->queue_id == 0) {
				cl->stats.rx ++;
			#if defined(FAST_STAT)
				cl->stats.rx_datalen += PKT_LEN;
			#else
				cl->stats.rx_datalen += pkts[i]->data_len;
			#endif
			}
		} else {
			onvm_pkt_drop(pkts[i]);
			cl->stats.rx_drop ++;
		}
#else
		/* Send packets to NF */
		onvm_pkt_enqueue_nf(rx, meta->destination, pkts[i]);
#endif /* BQUEUE_SWITCH */
	}

#if !defined(BQUEUE_SWITCH)
	onvm_pkt_flush_all_nfs(rx);
#endif
}


void
onvm_pkt_process_tx_batch(struct thread_info *tx, struct rte_mbuf *pkts[], uint16_t tx_count, struct client *cl) {
	uint16_t i;
	struct onvm_pkt_meta *meta;

	if (tx == NULL || pkts == NULL || cl == NULL)
		return;

	for (i = 0; i < tx_count; i++) {
		meta = (struct onvm_pkt_meta*) &(((struct rte_mbuf*)pkts[i])->udata64);
		meta->src = cl->info->service_id;

		/* The NF drops this packet explicitly */
		if (meta->action == ONVM_NF_ACTION_DROP) {
			/* If the packet is drop, then <return value> is 0, and !<return value> is 1. */
			cl->stats.act_drop += !onvm_pkt_drop(pkts[i]);
			continue;
		}

		meta->action = onvm_sc_next_action(default_chain, pkts[i]);
		meta->destination = onvm_sc_next_destination(default_chain, pkts[i]);

		(meta->chain_index) ++;

		/* NOTE: we remove ONVM_NF_ACTION_DROP, and make the NF drop the packet in the library by 
		 * freeing the rte_mbuf packet buffer. Now a NF do not need to specify how to process each
		 * packet because they should not know it except dropping the packet. The runtime system 
		 * should be responsible for determine the action of a packet because it knows the placement
		 * of the NFs. */

		if (meta->action == ONVM_NF_ACTION_TONF) {
			cl->stats.act_tonf++;
			onvm_pkt_enqueue_nf(tx, meta->destination, pkts[i]);
		} else if (meta->action == ONVM_NF_ACTION_OUT) {
			cl->stats.act_out++;
			onvm_pkt_enqueue_port(tx, meta->destination, pkts[i]);
		} else if (meta->action == ONVM_NF_ACTION_LOOP) {
			cl->stats.act_out++;
			/* To the start of the service chain, for test */
			meta->chain_index = 0;
			onvm_pkt_enqueue_nf(tx, meta->destination, pkts[i]);
		} else if (meta->action == ONVM_NF_ACTION_DROP) {
			/* The installed rule is to drop the packet, test only */
			cl->stats.act_drop += !onvm_pkt_drop(pkts[i]);
		} else {
			rte_exit(EXIT_FAILURE, "ERROR invalid action.\n");
			onvm_pkt_drop(pkts[i]);
			return;
		}
	}
}

void
onvm_pkt_bqueue_switch(struct thread_info *tx, struct client *cl) {
	struct onvm_pkt_meta *meta;
	struct rte_mbuf *pkt;
	int cnt = 0;
	unsigned int tx_queue_id = 0;
	int PKT_FWD_BATCH = 32;
	int res;

	for (;;) {
		res = bq_dequeue(cl->tx_bq[tx_queue_id], &pkt);
		if (res != SUCCESS) {
			tx_queue_id ++;
			if (tx_queue_id >= cl->gpu_info->thread_num) {
				tx_queue_id = 0;
			}
			continue;
		}
		cnt ++;

		meta = (struct onvm_pkt_meta*) &(pkt->udata64);
		meta->src = cl->info->service_id;

		/* The NF drops this packet explicitly */
		if (meta->action == ONVM_NF_ACTION_DROP) {
			/* If the packet is drop, then <return value> is 0, and !<return value> is 1. */
			cl->stats.act_drop += !onvm_pkt_drop(pkt);
			continue;
		}

		meta->action = onvm_sc_next_action(default_chain, pkt);
		meta->destination = onvm_sc_next_destination(default_chain, pkt);

		(meta->chain_index) ++;

		if (meta->action == ONVM_NF_ACTION_TONF) {
			cl->stats.act_tonf++;
			onvm_pkt_enqueue_nf(tx, meta->destination, pkt);
		} else if (meta->action == ONVM_NF_ACTION_OUT) {
			cl->stats.act_out++;
			onvm_pkt_enqueue_port(tx, meta->destination, pkt);
		} else if (meta->action == ONVM_NF_ACTION_LOOP) {
			cl->stats.act_out++;
			/* To the start of the service chain, for test */
			meta->chain_index = 0;
			onvm_pkt_enqueue_nf(tx, meta->destination, pkt);
		} else if (meta->action == ONVM_NF_ACTION_DROP) {
			/* The installed rule is to drop the packet, test only */
			cl->stats.act_drop += !onvm_pkt_drop(pkt);
		} else {
			rte_exit(EXIT_FAILURE, "ERROR invalid action.\n");
			onvm_pkt_drop(pkt);
			return;
		}

		if (cnt == PKT_FWD_BATCH) {
			cnt = 0;
			/* Dequeue from all threads in the NF */
			tx_queue_id ++;
			if (tx_queue_id >= cl->gpu_info->thread_num) {
				tx_queue_id = 0;
			}
		}
	}
}

void
onvm_pkt_flush_all_ports(struct thread_info *tx) {
	uint16_t i;

	if (tx == NULL)
		return;

	for (i = 0; i < ports->num_ports; i++)
		onvm_pkt_flush_port_queue(tx, i);
}


void
onvm_pkt_flush_all_nfs(struct thread_info *tx) {
	uint16_t i;

	if (tx == NULL)
		return;

	for (i = 0; i < MAX_CLIENTS; i++)
		onvm_pkt_flush_nf_queue(tx, i);
}

void
onvm_pkt_drop_batch(struct rte_mbuf **pkts, uint16_t size) {
	uint16_t i;

	if (pkts == NULL)
		return;

	for (i = 0; i < size; i++)
		rte_pktmbuf_free(pkts[i]);
}


/****************************Internal functions*******************************/


static void
onvm_pkt_flush_port_queue(struct thread_info *tx, uint16_t port) {
	uint16_t i, sent;
	volatile struct tx_stats *tx_stats;

	if (tx == NULL)
		return;

	if (tx->port_tx_buf[port].count == 0)
		return;

	tx_stats = &(ports->tx_stats);
	sent = rte_eth_tx_burst(port,
			tx->queue_id,
			tx->port_tx_buf[port].buffer,
			tx->port_tx_buf[port].count);
	if (unlikely(sent < tx->port_tx_buf[port].count)) {
		for (i = sent; i < tx->port_tx_buf[port].count; i++) {
			onvm_pkt_drop(tx->port_tx_buf[port].buffer[i]);
		}
		tx_stats->tx_drop[port] += (tx->port_tx_buf[port].count - sent);
	}
	tx_stats->tx[port] += sent;

	tx->port_tx_buf[port].count = 0;
}

static void
onvm_pkt_flush_nf_queue(struct thread_info *thread, uint16_t client) {
	uint16_t i;
	struct client *cl;
	struct rte_mbuf *pkt;

#if defined(BQUEUE_SWITCH)
	rte_exit(EXIT_FAILURE, "Should not enter this function with BQueue enabled\n");
#endif

	if (thread == NULL)
		return;

	if (thread->nf_rx_buf[client].count == 0)
		return;

	cl = &clients[client];

	// Ensure destination NF is running and ready to receive packets
	if (!onvm_nf_is_valid(cl))
		return;

	if (unlikely(cl->stats.reset == 1)) {
		cl->stats.reset = 0;
		cl->stats.rx = 0;
		cl->stats.rx_datalen = 0;
		cl->stats.rx_drop = 0;
		clock_gettime(CLOCK_MONOTONIC, &(cl->stats.start));
	}

#if defined(MATCH_RX_THREAD)
	if (unlikely(ONVM_NUM_RX_THREADS != cl->gpu_info->thread_num)) {
		printf("%d != %d\n", ONVM_NUM_RX_THREADS, cl->gpu_info->thread_num);
		assert(0);
	}
#endif

#if defined(MATCH_RX_THREAD) 
	if (rte_ring_enqueue_bulk(cl->rx_q[thread->queue_id], (void **)thread->nf_rx_buf[client].buffer,
				thread->nf_rx_buf[client].count, NULL) == 0) {
#else
	if (rte_ring_enqueue_bulk(cl->rx_q[cl->queue_id], (void **)thread->nf_rx_buf[client].buffer,
				thread->nf_rx_buf[client].count, NULL) == 0) {
#endif
		//RTE_LOG(DEBUG, APP, "NF RX queue full, unable to enqueue\n");

		for (i = 0; i < thread->nf_rx_buf[client].count; i++) {
			onvm_pkt_drop(thread->nf_rx_buf[client].buffer[i]);
		}
		cl->stats.rx_drop += thread->nf_rx_buf[client].count;

		/* TODO: If the queue is full, it means the corresponding NF CPU threads are incapable of processing
		 * the current network traffic. We launch new threads */
		//scheduler_nf_spawn_new_thread(cl);
	} else {
		cl->stats.rx += thread->nf_rx_buf[client].count;
		pkt = thread->nf_rx_buf[client].buffer[0];
		cl->stats.rx_datalen +=  pkt->data_len * thread->nf_rx_buf[client].count;
	}

	thread->nf_rx_buf[client].count = 0;

#if !defined(MATCH_RX_THREAD)
	cl->queue_id ++;
	/* >= is to avoid multiple RX threads modify cl->queue_id simultaneously */
	if (cl->queue_id >= cl->gpu_info->thread_num) {
		cl->queue_id = 0;
	}
#endif
}


inline static void
onvm_pkt_enqueue_port(struct thread_info *tx, uint16_t port, struct rte_mbuf *buf) {

	if (tx == NULL || buf == NULL)
		return;

	tx->port_tx_buf[port].buffer[tx->port_tx_buf[port].count++] = buf;
	if (tx->port_tx_buf[port].count == PACKET_READ_SIZE) {
		onvm_pkt_flush_port_queue(tx, port);
	}
}


/* Both RX threads and TX threads will enter this function.
 * 1) If BQUEUE_SWITCH is turned on, onvm_pkt_process_rx_batch() of each RX thread  will enqueue to the
 * corresponding NF thread directly, not calling this function. onvm_pkt_process_tx_batch() of each TX
 * thread will call this function to enqueue packets. Each TX thread is in charge of one NF.
 * FIXME: Not support more than one NFs send packets to the same NF.
 * 2) If BQUEUE_SWITCH is turned off, both RX threads and TX threads will call this function directly. */
inline static void
onvm_pkt_enqueue_nf(struct thread_info *thread, uint16_t dst_service_id, struct rte_mbuf *pkt) {
	struct client *cl;
	uint16_t dst_instance_id;

	if (thread == NULL || pkt == NULL)
		return;

	// map service to instance and check one exists
	dst_instance_id = onvm_nf_service_to_nf_map(dst_service_id, pkt);
	if (dst_instance_id == 0) {
		onvm_pkt_drop(pkt);
		return;
	}

	// Ensure destination NF is running and ready to receive packets
	cl = &clients[dst_instance_id];
	if (!onvm_nf_is_valid(cl)) {
		onvm_pkt_drop(pkt);
		return;
	}

#if defined(BQUEUE_SWITCH)
	if (unlikely(cl->stats.reset == 1)) {
		cl->stats.reset = 0;
		cl->stats.rx = 0;
		cl->stats.rx_datalen = 0;
		cl->stats.rx_drop = 0;
		clock_gettime(CLOCK_MONOTONIC, &(cl->stats.start));
	}

	if (SUCCESS == bq_enqueue(cl->rx_bq[cl->queue_id], pkt)) {
		cl->stats.rx ++;
	#if defined(FAST_STAT)
		cl->stats.rx_datalen += PKT_LEN;
	#else
		cl->stats.rx_datalen += pkt->data_len;
	#endif
	} else {
		onvm_pkt_drop(pkt);
		cl->stats.rx_drop ++;
	}

	cl->queue_id ++;
	if (cl->queue_id >= cl->gpu_info->thread_num) {
		cl->queue_id = 0;
	}
#else
	thread->nf_rx_buf[dst_instance_id].buffer[thread->nf_rx_buf[dst_instance_id].count++] = pkt;
	if (thread->nf_rx_buf[dst_instance_id].count == PACKET_READ_SIZE) {
		onvm_pkt_flush_nf_queue(thread, dst_instance_id);
	}
#endif
}


inline static void
onvm_pkt_process_next_action(struct thread_info *tx, struct rte_mbuf *pkt, struct client *cl) {

	if (tx == NULL || pkt == NULL || cl == NULL)
		return;

	struct onvm_flow_entry *flow_entry;
	struct onvm_service_chain *sc;
	struct onvm_pkt_meta *meta = onvm_get_pkt_meta(pkt);
	int ret;

	ret = onvm_flow_dir_get_pkt(pkt, &flow_entry);
	if (ret >= 0) {
		sc = flow_entry->sc;
		meta->action = onvm_sc_next_action(sc, pkt);
		meta->destination = onvm_sc_next_destination(sc, pkt);
	} else {
		meta->action = onvm_sc_next_action(default_chain, pkt);
		meta->destination = onvm_sc_next_destination(default_chain, pkt);
	}

	switch (meta->action) {
		case ONVM_NF_ACTION_DROP:
			// if the packet is drop, then <return value> is 0
			// and !<return value> is 1.
			cl->stats.act_drop += !onvm_pkt_drop(pkt);
			break;
		case ONVM_NF_ACTION_TONF:
			cl->stats.act_tonf++;
			onvm_pkt_enqueue_nf(tx, meta->destination, pkt);
			break;
		case ONVM_NF_ACTION_OUT:
			cl->stats.act_out++;
			onvm_pkt_enqueue_port(tx, meta->destination, pkt);
			break;
		default:
			break;
	}
	(meta->chain_index)++;
}


/*******************************Helper function*******************************/


static int
onvm_pkt_drop(struct rte_mbuf *pkt) {
	rte_pktmbuf_free(pkt);
	if (pkt != NULL) {
		return 1;
	}
	return 0;
}
