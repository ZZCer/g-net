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

                                 onvm_pkt.h


      Header file containing all prototypes of packet processing functions


******************************************************************************/


#ifndef _ONVM_PKT_H_
#define _ONVM_PKT_H_

#include <rte_mbuf.h>

/*********************************Interfaces**********************************/


/*
 * Interface to process packets in a given RX queue.
 *
 * Inputs : a pointer to the rx queue
 *          an array of packets
 *          the size of the array
 *
 */
void
onvm_pkt_process_rx_batch(struct thread_info *rx, struct rte_mbuf *pkts[], uint16_t rx_count);


/*
 * Interface to process packets in a given TX queue.
 *
 * Inputs : a pointer to the tx queue
 *          an array of packets
 *          the size of the array
 *          a pointer to the client possessing the TX queue.
 *
 */
void
onvm_pkt_process_tx_batch(struct thread_info *tx, struct rte_mbuf *pkts[], uint16_t tx_count, struct client *cl);


/* Switching with B-Queue */
void
onvm_pkt_bqueue_switch(struct thread_info *tx, struct client *cl);


/*
 * Interface to send packets to all ports after processing them.
 *
 * Input : a pointer to the tx queue
 *
 */
void
onvm_pkt_flush_all_ports(struct thread_info *tx);


/*
 * Interface to send packets to all NFs after processing them.
 *
 * Input : a pointer to the tx queue
 *
 */
void
onvm_pkt_flush_all_nfs(struct thread_info *tx);


/*
 * Interface to drop a batch of packets.
 *
 * Inputs : the array of packets
 *          the size of the array
 *
 */
void
onvm_pkt_drop_batch(struct rte_mbuf **pkts, uint16_t size);


void
onvm_pkt_flush_port_queue(struct thread_info *tx, uint16_t port);


inline static void
onvm_pkt_drop(struct rte_mbuf *pkt) {
	rte_pktmbuf_free(pkt);
}

inline static void
onvm_pkt_drop_batch(struct rte_mbuf **pkts, uint16_t size) {
	uint16_t i;

	if (pkts == NULL)
		return;

	for (i = 0; i < size; i++)
		onvm_pkt_drop(pkts[i]);
}

#endif  // _ONVM_PKT_H_
