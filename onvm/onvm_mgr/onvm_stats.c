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
}


void
onvm_stats_clear_all_clients(void) {
	unsigned i;

	for (i = 0; i < MAX_CLIENTS; i++) {
		clients[i].stats.rx = clients[i].stats.rx_drop = 0;
		clients[i].stats.rx_datalen = 0;
		clients[i].stats.act_drop = clients[i].stats.act_tonf = 0;
		clients[i].stats.act_next = clients[i].stats.act_out = 0;
		clients[i].stats.reset = 0;
	}
}

void
onvm_stats_clear_client(uint16_t id) {
	clients[id].stats.rx = clients[id].stats.rx_drop = 0;
	clients[id].stats.rx_datalen = 0;
	clients[id].stats.act_drop = clients[id].stats.act_tonf = 0;
	clients[id].stats.act_next = clients[id].stats.act_out = 0;
	clients[id].stats.reset = 0;
}


/****************************Internal functions*******************************/


static void
onvm_stats_display_ports(unsigned difftime) {
	unsigned i;
	/* Arrays to store last TX/RX count to calculate rate */
	static uint64_t tx_last[RTE_MAX_ETHPORTS];
	static uint64_t rx_last[RTE_MAX_ETHPORTS];

	printf("PORTS\n");
	printf("-----\n");
	for (i = 0; i < ports->num_ports; i++)
		printf("Port %u: '%s'\t", (unsigned)ports->id[i],
				onvm_stats_print_MAC(ports->id[i]));
	printf("\n\n");
	for (i = 0; i < ports->num_ports; i++) {
		printf("Port %u - rx: %9"PRIu64"  (%9"PRIu64" pps)\t"
				"tx: %9"PRIu64"  (%9"PRIu64" pps)\n",
				(unsigned)ports->id[i],
				ports->rx_stats.rx[ports->id[i]],
				(ports->rx_stats.rx[ports->id[i]] - rx_last[i])
				/difftime,
				ports->tx_stats.tx[ports->id[i]],
				(ports->tx_stats.tx[ports->id[i]] - tx_last[i])
				/difftime);

		rx_last[i] = ports->rx_stats.rx[ports->id[i]];
		tx_last[i] = ports->tx_stats.tx[ports->id[i]];
	}

	/* Hardware statistic */
	static struct timespec start, end;
	struct rte_eth_stats stats;
	static uint64_t ibytes = 0, obytes = 0;
	static uint64_t ipackets = 0, opackets = 0;

	clock_gettime(CLOCK_MONOTONIC, &end);
	double diff = 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000;

	rte_eth_stats_get(0, &stats);
	printf("\n===== HW statistics for Port 0: err packets-%lu,\treceived packets-%lu,\ttransmitted packets-%lu,\tdropped packets-%lu\n"
			"\tInput Speed - [[[ %.2lf Mpps, %.2lf Gbps ]]], Output Speed - [[[ %.2lf Mpps, %.2lf Gbps ]]]\n",
			stats.ierrors, stats.ipackets, stats.opackets, stats.imissed,
			(double)(stats.ipackets-ipackets)/diff, ((double)(stats.ibytes-ibytes) * 8) / (1000 * diff),
			(double)(stats.opackets-opackets)/diff, ((double)(stats.obytes-obytes) * 8) / (1000 * diff));

	ibytes = stats.ibytes;
	obytes = stats.obytes;
	ipackets = stats.ipackets;
	opackets = stats.opackets;

#if defined(RX_SPEED_TEST) || defined(RX_SPEED_TEST_2)
	printf("\n===== RX SPEED TEST =====\n");
	uint64_t rx_count = 0, rx_bytes = 0;

	for (i = 0; i < ONVM_NUM_RX_THREADS; i ++) {
		rx_count += rx_stats[i].count;
		rx_bytes += rx_stats[i].bytes;

		printf("\tRX queue %d: rx %lu pkts, speed %.2lf Mpps, %.2lf Gbps\n",
				i, rx_stats[i].count, (double)rx_stats[i].count/diff, (double)((rx_stats[i].bytes + 20 * rx_stats[i].count) * 8) / (1000 * diff));

		rx_stats[i].reset = 1;
	}

	printf("\tRX Overall Performance: %.2lf Mpps, %.2lf Gbps\n", (double)rx_count/diff, (double)((rx_bytes + 20 * rx_count) * 8) / (1000 * diff));
#endif

	clock_gettime(CLOCK_MONOTONIC, &start);
}


static void
onvm_stats_display_clients(void) {
	unsigned i;
	static struct timespec end;
	static uint64_t tx_last[MAX_CLIENTS];
	static uint64_t nf_drop_last[MAX_CLIENTS], nf_drop_l;
	static uint64_t nf_drop_enq_last[MAX_CLIENTS], nf_drop_enq_l;
	double diff;

	printf("\nCLIENTS\n");
	printf("-------\n");
	for (i = 0; i < MAX_CLIENTS; i++) {
		if (!onvm_nf_is_valid(&clients[i]))
			continue;

		/* avoid dividing 0 */
		if (clients[i].stats.rx == 0)
			clients[i].stats.rx = 1;

		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = 1000000 * (end.tv_sec - clients[i].stats.start.tv_sec)
			+ (end.tv_nsec - clients[i].stats.start.tv_nsec) / 1000;

#if defined(BQUEUE_SWITCH)
		uint64_t rx;
		uint64_t rx_datalen;
		if (clients[i].instance_id == 1) {
			rx = clients[i].stats.rx * ONVM_NUM_RX_THREADS;
			rx_datalen = clients[i].stats.rx_datalen * ONVM_NUM_RX_THREADS;
		} else {
			rx = clients[i].stats.rx;
			rx_datalen = clients[i].stats.rx_datalen;
		}
#else
		const uint64_t rx = clients[i].stats.rx;
		const uint64_t rx_datalen = clients[i].stats.rx_datalen;
#endif
		const uint64_t rx_drop = clients[i].stats.rx_drop;
		const uint64_t act_drop = clients[i].stats.act_drop;
		const uint64_t act_next = clients[i].stats.act_next;
		const uint64_t act_out = clients[i].stats.act_out;
		const uint64_t act_tonf = clients[i].stats.act_tonf;
		const uint64_t tx = clients_stats[i].tx * clients[i].gpu_info->thread_num;
		const uint64_t tx_drop = clients_stats[i].tx_drop * clients[i].gpu_info->thread_num;
		const uint64_t nf_drop = clients_stats[i].nf_drop * clients[i].gpu_info->thread_num;
		const uint64_t nf_drop_enq = clients_stats[i].nf_drop_enq * clients[i].gpu_info->thread_num;
		const uint64_t batch_size = clients_stats[i].batch_size;
		const uint64_t batch_cnt = clients_stats[i].batch_cnt;

		nf_drop_l = nf_drop - nf_drop_last[i];
		nf_drop_enq_l = nf_drop_enq - nf_drop_enq_last[i];

		/* Update scheduler info */
		clients[i].avg_pkt_len = (double)rx_datalen / rx;
		clients[i].throughput_mpps = (double)(rx - nf_drop_l) / diff;
		//clients[i].throughput_mpps = (double)(rx) / diff;
		clients[i].stats.reset = 1;

		printf("\n[Client %u - %s] :\nrx: %9"PRIu64"\trx_drop: %9"PRIu64"\tnext: %9"PRIu64"\tdrop: %9"PRIu64"\tnf_drop: %9"PRIu64"\tnf_drop_enq: %9"PRIu64"\t[[[ %.6lf Mpps, %.6lf Gbps, Pkt size %ld ]]]\n"
				"tx: %9"PRIu64"\ttx_drop: %9"PRIu64"\tout:  %9"PRIu64"\ttonf: %9"PRIu64"\t[[[ %.6lf Mpps ]]]\n",
				clients[i].instance_id, get_nf_name(clients[i].info->service_id),
				rx, rx_drop, act_next, act_drop, nf_drop_l, nf_drop_enq_l,
				(double)(rx - nf_drop_l)/diff, (double)((rx - nf_drop_l) * (rx_datalen/rx + 20) * 8)/(1000 * diff), rx_datalen/rx,
				tx, tx_drop, act_out, act_tonf, (double)(tx-tx_last[i])/diff);

		printf("GPU: Average batch size per thread is %.2lf, set batch as %d\n", (double)batch_size/batch_cnt, clients[i].batch_size);
		if (clients[i].stats.batch_cnt == 0) {
			printf("Kernel count is 0, no statistics\n");
		} else {
			printf("Average HtoD PCIe size each kernel is %ld bytes, DtoH PCIe size is %ld bytes, batch count is %ld\n", clients[i].stats.htod_mem/clients[i].stats.batch_cnt, clients[i].stats.dtoh_mem/clients[i].stats.batch_cnt, clients[i].stats.batch_cnt);
		}
		clients[i].stats.htod_mem = 0;
		clients[i].stats.dtoh_mem = 0;
		clients[i].stats.batch_cnt = 0;

	#if defined(MEASURE_KERNEL_TIME)
		printf("Average kernel execution time in Manager: %.2lf us\n", total_kernel_time_diff/total_kernel_time_cnt); 
		total_kernel_time_cnt = 0;
		total_kernel_time_diff = 0;
	#endif

		printf("Average GPU execution time (kernel + PCIe + message) in NF: %.2lf us, batch count is %ld\n", clients_stats[i].gpu_time/clients_stats[i].gpu_time_cnt, clients_stats[i].gpu_time_cnt);

		tx_last[i] = tx;
		nf_drop_last[i] = nf_drop;
		nf_drop_enq_last[i] = nf_drop_enq;

		if (clients[i].info->service_id == NF_PKTGEN) {
			clock_gettime(CLOCK_MONOTONIC, &(clients[i].stats.start));
		}
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
