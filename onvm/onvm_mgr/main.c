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


#include "onvm_mgr.h"
#include "onvm_pkt.h"
#include "onvm_nf.h"
#include "onvm_stats.h"
#include "onvm_init.h"
#include "manager.h"
#include "scheduler.h"

extern struct onvm_service_chain *default_chain;
extern struct rx_perf rx_stats[ONVM_NUM_RX_THREADS]; 

/*******************************Worker threads********************************/

/*
 * Function to receive packets from the NIC
 * and distribute them to the default service
 */
static int
rx_thread_main(void *arg) {
	uint16_t i, rx_count;
	struct rte_mbuf *pkts[PACKET_READ_SIZE];
	struct thread_info *rx = (struct thread_info*)arg;
	unsigned int core_id = rte_lcore_id();

	RTE_LOG(INFO, APP, "Core %d: Running RX thread for RX queue %d\n", core_id, rx->queue_id);
	
#if defined(RX_SPEED_TEST)
	uint16_t j;
	rx_stats[rx->queue_id].count = 0;
	rx_stats[rx->queue_id].bytes = 0;
#endif

	for (;;) {
		/* Read ports */
		for (i = 0; i < ports->num_ports; i++) {
			rx_count = rte_eth_rx_burst(ports->id[i], rx->queue_id, pkts, PACKET_READ_SIZE);
			ports->rx_stats.rx[ports->id[i]] += rx_count;

#if defined(RX_SPEED_TEST) || defined(RX_SPEED_TEST_2)
			if (unlikely(rx_count == 0)) continue;

			if (unlikely(rx_stats[rx->queue_id].reset == 1)) {
				rx_stats[rx->queue_id].count = 0;
				rx_stats[rx->queue_id].bytes = 0;
				rx_stats[rx->queue_id].reset = 0;
			}

			rx_stats[rx->queue_id].count += rx_count;
			rx_stats[rx->queue_id].bytes += rx_count * pkts[0]->data_len;
#endif

#if defined(RX_SPEED_TEST)
			for (j = 0; j < rx_count; j ++) {
				rte_pktmbuf_free(pkts[j]);
			}
			continue;
#endif

			/* Now process the NIC packets read */
			if (likely(rx_count > 0)) {
				// If there is no running NF, we drop all the packets of the batch.
				if (!num_clients) {
					onvm_pkt_drop_batch(pkts, rx_count);
				} else {
					onvm_pkt_process_rx_batch(rx, pkts, rx_count);
				}
			}
		}
	}

	return 0;
}

#ifndef NEW_SWITCHING
static int
tx_thread_main(void *arg) {
	struct thread_info *tx = (struct thread_info*)arg;
	struct client *cl = &(clients[tx->first_cl]);
#if !defined(BQUEUE_SWITCH)
	unsigned tx_count;
	struct rte_mbuf *pkts[PACKET_READ_SIZE];
#endif

	RTE_LOG(INFO, APP, "Core %d: Running TX thread for NF %d\n", rte_lcore_id(), tx->first_cl);

	for (;;) {
		/* Read packets from the client's tx queue and process them as needed */
		if (!onvm_nf_is_valid(cl))
			continue;

#if defined(BQUEUE_SWITCH)
		onvm_pkt_bqueue_switch(tx, cl);
#else
		/* Dequeue all packets in ring up to max possible. */
		tx_count = rte_ring_dequeue_burst(cl->tx_q, (void **)pkts, PACKET_READ_SIZE);

		/* Now process the Client packets read */
		if (likely(tx_count > 0)) {
			onvm_pkt_process_tx_batch(tx, pkts, tx_count, cl);
		}

		/* Send a burst to every port */
		onvm_pkt_flush_all_ports(tx);

		/* Send a burst to every NF */
		onvm_pkt_flush_all_nfs(tx);
#endif
	}

	return 0;
}
#else // NEW_SWITCHING
static int
tx_thread_main(void *arg) {
	struct thread_info *tx = (struct thread_info*)arg;
	unsigned int core_id = rte_lcore_id();
	int port_id = tx->queue_id;

	RTE_LOG(INFO, APP, "Core %d: Running TX thread for port %d\n", core_id, port_id);



	return 0;
}
#endif // NEW_SWITCHING

/*******************************Main function*********************************/


int
main(int argc, char *argv[]) {
	unsigned cur_lcore, rx_lcores, tx_lcores;
	unsigned i;

	/* initialise the system */

	/* Reserve ID 0 for internal manager things */
	next_instance_id = 1;
	if (init(argc, argv) < 0)
		return -1;
	RTE_LOG(INFO, APP, "Finished Process Init.\n");

	/*
	const struct rte_memseg *seg = rte_eal_get_physmem_layout();
	for (i = 0; ; i ++) { 
		if (seg[i].addr == NULL) break;
		RTE_LOG(INFO, APP, "virtual address [%p, %p], physical address %p, len %zu\n",
				seg[i].addr, (uint8_t *)seg[i].addr + seg[i].len, (void *)seg[i].phys_addr, seg[i].len);
	}
	*/

	/* clear statistics */
	onvm_stats_clear_all_clients();

	/* Reserve n cores for: 1 Scheduler + Stats, 1 Manager, and ONVM_NUM_RX_THREADS for Rx */
	cur_lcore = rte_lcore_id();
	rx_lcores = ONVM_NUM_RX_THREADS;
#if defined(NETWORK)
	tx_lcores = default_chain->chain_length - 1;
#else
	tx_lcores = default_chain->chain_length; /* equal number of VMs, considering NF_PKTGEN */
#endif

	RTE_LOG(INFO, APP, "%d cores available in total\n", rte_lcore_count());
	RTE_LOG(INFO, APP, "%d cores available for handling RX queues\n", rx_lcores);
	RTE_LOG(INFO, APP, "%d cores available for handling TX queues\n", tx_lcores);
	RTE_LOG(INFO, APP, "%d cores available for Manager\n", 1);
	RTE_LOG(INFO, APP, "%d cores available for Scheduler + States\n", 1);

	if (rx_lcores + tx_lcores + 2 != rte_lcore_count()) {
		rte_exit(EXIT_FAILURE, "%d cores needed, but there are only %d cores specified\n", rx_lcores+tx_lcores+2, rte_lcore_count());
	}

	// We start the system with 0 NFs active
	num_clients = 0;

	/* Launch Manager thread as the GPU proxy */
	cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
	if (rte_eal_remote_launch(manager_thread_main, NULL, cur_lcore) == -EBUSY) {
		RTE_LOG(ERR, APP, "Core %d is already busy, can't use for Manager\n", cur_lcore);
		return -1;
	}

#ifndef NEW_SWITCHING
	/* Assign each NF with a TX thread */
	for (i = 0; i < tx_lcores; i++) {
		struct thread_info *tx = calloc(1, sizeof(struct thread_info));
		tx->queue_id = i; /* FIXME: This way of setting queue_id is wrong, as only the last NF in the service chain can use the tx queue in the NIC */
		tx->port_tx_buf = calloc(RTE_MAX_ETHPORTS, sizeof(struct packet_buf));
		tx->nf_rx_buf = calloc(MAX_CLIENTS, sizeof(struct packet_buf));
		tx->first_cl = i + 1;

		cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
		if (rte_eal_remote_launch(tx_thread_main, (void*)tx,  cur_lcore) == -EBUSY) {
			RTE_LOG(ERR, APP, "Core %d is already busy, can't use for client %d TX\n", cur_lcore, tx->first_cl);
			return -1;
		}
	}
#else
	for (i = 0; i < ports->num_ports; i++) {
		struct thread_info *tx = calloc(1, sizeof(struct thread_info));
		tx->queue_id = ports->id[i]; /* Actually this is the port id */

		cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
		if (rte_eal_remote_launch(tx_thread_main, (void*)tx,  cur_lcore) == -EBUSY) {
			RTE_LOG(ERR, APP, "Core %d is already busy, can't use for port %d TX\n", cur_lcore, tx->queue_id);
			return -1;
		}
	}
#endif

	/* Launch RX thread main function for each RX queue on cores */
	for (i = 0; i < rx_lcores; i++) {
		struct thread_info *rx = calloc(1, sizeof(struct thread_info));
		rx->queue_id = i;
		rx->port_tx_buf = NULL;
		rx->nf_rx_buf = calloc(MAX_CLIENTS, sizeof(struct packet_buf));

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
