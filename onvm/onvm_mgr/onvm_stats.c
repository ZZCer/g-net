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
                          onvm_stats.c

   This file contain the implementation of all functions related to
   statistics display in the manager.

******************************************************************************/


#include "onvm_mgr.h"
#include "onvm_stats.h"
#include "onvm_nf.h"
#include "onvm_common.h"
#include "onvm_init.h"


struct rx_perf rx_stats[ONVM_NUM_RX_THREADS]; 

extern double total_kernel_time_diff;
extern uint32_t total_kernel_time_cnt;

/************************Internal Functions Prototypes************************/


/*
 * Function displaying statistics for all ports
 *
 * Input : time passed since last display (to compute packet rate)
 *
 */
static void
onvm_stats_display_ports(unsigned difftime);


/*
 * Function displaying statistics for all clients
 *
 */
static void
onvm_stats_display_clients(void);


/*
 * Function clearing the terminal and moving back the cursor to the top left.
 * 
 */
static void
onvm_stats_clear_terminal(void);


/*
 * Function giving the MAC address of a port in string format.
 *
 * Input  : port
 * Output : its MAC address
 * 
 */
static const char *
onvm_stats_print_MAC(uint8_t port);


/****************************Interfaces***************************************/


void
onvm_stats_display_all(unsigned difftime) {
	onvm_stats_clear_terminal();
	onvm_stats_display_ports(difftime);
	onvm_stats_display_clients();
	struct rte_mempool *mp;
	mp = rte_mempool_lookup(PKTMBUF_POOL_NAME);
	printf("AVAIL MBUFS: %u\n", rte_mempool_avail_count(mp));
}

void
onvm_stats_clear_client(uint16_t i) {
	rte_spinlock_lock(&clients[i].stats.update_lock);
	clock_gettime(CLOCK_MONOTONIC, &clients[i].stats.start);
	clients[i].stats.rx = 0;
	clients[i].stats.rx_datalen = 0;
	clients[i].stats.tx = 0;
	clients[i].stats.tx_drop = 0;
	clients[i].stats.act_drop = 0;
	clients[i].stats.htod_mem = 0;
	clients[i].stats.dtoh_mem = 0;
	clients[i].stats.batch_size = 0;
	clients[i].stats.batch_cnt = 0;
	clients[i].stats.cpu_time = 0;
	clients[i].stats.gpu_time = 0;
	clients[i].stats.kernel_time = 0;
	clients[i].stats.kernel_cnt = 0;
	rte_spinlock_unlock(&clients[i].stats.update_lock);
}

void
onvm_stats_clear_all_clients(void) {
	unsigned i;

	for (i = 0; i < MAX_CLIENTS; i++) {
		onvm_stats_clear_client(i);
	}
}


/****************************Internal functions*******************************/

#define PKT_LEN_ETH_SPEC 20

static void
onvm_stats_display_ports(unsigned difftime) {
	unsigned i;
	/* Arrays to store last TX/RX count to calculate rate */
	static uint64_t tx_last[RTE_MAX_ETHPORTS] = {0}, tx_len_last[RTE_MAX_ETHPORTS] = {0};
	static uint64_t rx_last[RTE_MAX_ETHPORTS] = {0}, rx_len_last[RTE_MAX_ETHPORTS] = {0};
	static uint64_t rx_gpucopy_last = 0, rx_len_gpucopy_last = 0;
	/* Hardware statistic */
	static struct timespec start, end;
	struct rte_eth_stats stats;
	static uint64_t ibytes[RTE_MAX_ETHPORTS] = {0}, obytes[RTE_MAX_ETHPORTS] = {0};
	static uint64_t ipackets[RTE_MAX_ETHPORTS] = {0}, opackets[RTE_MAX_ETHPORTS] = {0};

	printf("PORTS\n");
	printf("-----\n");
	for (i = 0; i < ports->num_ports; i++)
		printf("Port %u: '%s'\t", (unsigned)ports->id[i],
				onvm_stats_print_MAC(ports->id[i]));
	printf("\n\n");
	for (i = 0; i < ports->num_ports; i++) {
		uint64_t rx_count = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx[ports->id[i]]);
		uint64_t rx_len = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx_len[ports->id[i]]);
		uint64_t rx_gpucopy = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx_gpucopy);
		uint64_t rx_len_gpucopy = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->rx_stats.rx_len_gpucopy);
		uint64_t tx_count = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->tx_stats.tx[ports->id[i]]);
		uint64_t tx_len = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->tx_stats.tx_len[ports->id[i]]);

		printf("Port %u - \trx:\t %9"PRIu64"  (%9"PRIu64" pps, %.6lf Gbps)\n"
				"\t\tgpucopy: %9"PRIu64"  (%9"PRIu64" pps, %.6lf Gbps)\n"
				"\t\ttx:\t %9"PRIu64"  (%9"PRIu64" pps, %.6lf Gbps)\n",
				(unsigned)ports->id[i],
				rx_count,
				(rx_count - rx_last[i]) / difftime,
				(double)(rx_len - rx_len_last[i] + (rx_count - rx_last[i]) * PKT_LEN_ETH_SPEC) / 1e9 * 8 
				/difftime,

				rx_gpucopy,
				(rx_gpucopy - rx_gpucopy_last) / difftime,
				(double)(rx_len_gpucopy - rx_len_gpucopy_last + (rx_gpucopy - rx_gpucopy_last) * PKT_LEN_ETH_SPEC) / 1e9 * 8 
				/difftime,

				tx_count,
				(tx_count - tx_last[i])
				/difftime,
				(double)(tx_len - tx_len_last[i] + (tx_count - tx_last[i]) * PKT_LEN_ETH_SPEC) / 1e9 * 8 
				/difftime);

		printf("Tx GPU Batch: cnt %9"PRIu64", pkt %9"PRIu64", avg pkt %f\n",
				ports->tx_stats.gpu_batch_cnt[ports->id[i]],
				ports->tx_stats.gpu_batch_pkt[ports->id[i]],
				ports->tx_stats.gpu_batch_pkt[ports->id[i]] * 1.0 / ports->tx_stats.gpu_batch_cnt[ports->id[i]]
				);

		rx_last[i] = rx_count;
		rx_len_last[i] = rx_len;
		rx_gpucopy_last = rx_gpucopy;
		rx_len_gpucopy_last = rx_len_gpucopy;
		tx_last[i] = tx_count;
		tx_len_last[i] = tx_len;
		ports->tx_stats.gpu_batch_cnt[ports->id[i]] = 0;
		ports->tx_stats.gpu_batch_pkt[ports->id[i]] = 0;

		clock_gettime(CLOCK_MONOTONIC, &end);
		double diff = 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000;

		rte_eth_stats_get(0, &stats);
		printf("HW===\terr packets-%lu,\treceived packets-%lu,\ttransmitted packets-%lu,\tdropped packets-%lu\n"
				"\tInput Speed - [[[ %.2lf Mpps, %.2lf Gbps ]]], Output Speed - [[[ %.2lf Mpps, %.2lf Gbps ]]]\n",
				stats.ierrors, stats.ipackets, stats.opackets, stats.imissed,
				(double)(stats.ipackets-ipackets[ports->id[i]])/diff, ((double)(stats.ibytes-ibytes[ports->id[i]]) * 8) / (1000 * diff),
				(double)(stats.opackets-opackets[ports->id[i]])/diff, ((double)(stats.obytes-obytes[ports->id[i]]) * 8) / (1000 * diff));

		ibytes[ports->id[i]] = stats.ibytes;
		obytes[ports->id[i]] = stats.obytes;
		ipackets[ports->id[i]] = stats.ipackets;
		opackets[ports->id[i]] = stats.opackets;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);
}


static void
onvm_stats_display_clients(void) {
	unsigned i;
	static struct timespec end;
	double diff;

	printf("\nCLIENTS\n");
	printf("-------\n");
	for (i = 0; i < MAX_CLIENTS; i++) {
		if (!onvm_nf_is_valid(&clients[i]))
			continue;

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = 1000000 * (end.tv_sec - clients[i].stats.start.tv_sec)
			+ (end.tv_nsec - clients[i].stats.start.tv_nsec) / 1000;

		uint64_t rx = clients[i].stats.rx;
		uint64_t rx_datalen = clients[i].stats.rx_datalen;
		uint64_t tx = clients[i].stats.tx;
		uint64_t tx_drop = clients[i].stats.tx_drop;
		uint64_t act_drop = clients[i].stats.act_drop;
		uint64_t batch_size = clients[i].stats.batch_size;
		uint64_t batch_cnt = clients[i].stats.batch_cnt;

		if (rx == 0) rx = 1;

		/* Update scheduler info */
		clients[i].avg_pkt_len = (double)rx_datalen / rx;
		clients[i].throughput_mpps = (double)(tx + tx_drop + act_drop) / diff;

		double approx_gbps = (double)((tx + tx_drop + act_drop) * (clients[i].avg_pkt_len + PKT_LEN_ETH_SPEC) * 8)/(1000 * diff);

		printf("\n[Client %u - %s] :\n"
		       "rx: %9lu\t" "tx: %9lu\t" "tx_drop: %9lu\t" "act_drop: %9lu\n"
			   "[[[ %.6lf Mpps, %.6lf Gbps, Pkt size %ld ]]]\n",
				clients[i].instance_id, get_nf_name(clients[i].info->service_id),
				rx, tx, tx_drop, act_drop,
				clients[i].throughput_mpps, approx_gbps, rx_datalen/rx);

		printf("GPU: Avg. batch size = %.2lf, => %d\n", (double)batch_size/batch_cnt, clients[i].batch_size);
		if (clients[i].stats.batch_cnt == 0) {
			printf("Kernel count is 0, no statistics\n");
		} else {
			printf("Avg HtoD = %ld bytes, DtoH = %ld bytes, Batch count = %ld\n",
					clients[i].stats.htod_mem/clients[i].stats.batch_cnt,
					clients[i].stats.dtoh_mem/clients[i].stats.batch_cnt,
					clients[i].stats.batch_cnt);
		}

		printf("Average GPU execution time in NF: %.2lf us (kernel:  %.2lf us, overhead: %.2lf us), CPU: %.2lf us, batch count is %ld\n",
				clients[i].stats.gpu_time/clients[i].stats.batch_cnt,
				clients[i].stats.kernel_time/clients[i].stats.batch_cnt,
				clients[i].stats.gpu_time/clients[i].stats.batch_cnt - clients[i].stats.kernel_time/clients[i].stats.batch_cnt,
				clients[i].stats.cpu_time/clients[i].stats.batch_cnt,
				clients[i].stats.batch_cnt);
	}

	printf("\n");
}


/***************************Helper functions**********************************/


static void
onvm_stats_clear_terminal(void) {
	const char clr[] = { 27, '[', '2', 'J', '\0' };
	const char topLeft[] = { 27, '[', '1', ';', '1', 'H', '\0' };

	printf("%s%s", clr, topLeft);
}


static const char *
onvm_stats_print_MAC(uint8_t port) {
	static const char err_address[] = "00:00:00:00:00:00";
	static char addresses[RTE_MAX_ETHPORTS][sizeof(err_address)];

	if (unlikely(port >= RTE_MAX_ETHPORTS))
		return err_address;
	if (unlikely(addresses[port][0] == '\0')) {
		struct ether_addr mac;
		rte_eth_macaddr_get(port, &mac);
		snprintf(addresses[port], sizeof(addresses[port]),
				"%02x:%02x:%02x:%02x:%02x:%02x\n",
				mac.addr_bytes[0], mac.addr_bytes[1],
				mac.addr_bytes[2], mac.addr_bytes[3],
				mac.addr_bytes[4], mac.addr_bytes[5]);
	}
	return addresses[port];
}
