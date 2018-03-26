/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2016 George Washington University
 *            2015-2016 University of California Riverside
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
 * speed_tester.c - create pkts and loop through NFs.
 ********************************************************************/

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_mempool.h>
#include <rte_cycles.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#define NF_TAG "pktgen"

#define NUM_PKTS 128 
#define PKTGEN_LOOP_TIME 6
#define PKTMBUF_POOL_NAME "MProc_pktmbuf_pool"

#define PKTGEN_MAC_ADDR	0
#define PKTGEN_IP_ADDR	1234
#define PKTGEN_UDP_PORT	567
#define PKTGEN_IP_PKT_LENGTH 256


#define KEEP_SENDING 0

/* number of package between each print */
static uint32_t print_delay = 10000000;
struct rte_mbuf *pkts[NUM_PKTS];

/*
 * This function displays stats. It uses ANSI terminal codes to clear
 * screen when called. It is called from a single non-master
 * thread in the server process, when the process is run with more
 * than one lcore enabled.
 */
static void
do_stats_display(void) {
	static uint64_t last_cycles;
	static uint64_t cur_pkts = 0;
	static uint64_t last_pkts = 0;
	const char clr[] = { 27, '[', '2', 'J', '\0' };
	const char topLeft[] = { 27, '[', '1', ';', '1', 'H', '\0' };

	uint64_t cur_cycles = rte_get_tsc_cycles();
	cur_pkts += print_delay;

	/* Clear screen and move to top left */
	printf("%s%s", clr, topLeft);

	printf("Total packets: %9"PRIu64" \n", cur_pkts);
	printf("TX pkts per second: %9"PRIu64" pps\n", (cur_pkts - last_pkts)
			* rte_get_timer_hz() / (cur_cycles - last_cycles));
	printf("\n\n");

	last_pkts = cur_pkts;
	last_cycles = cur_cycles;
}

#if !defined(KEEP_SENDING)
static int
loop_forward(struct rte_mbuf **recv_pkts, int pkt_num, int thread_id)
{
	UNUSED(thread_id);
	static uint32_t counter = 0;

	if (pkt_num == 0)
		return 0;

	if (counter++ == print_delay) {
		do_stats_display();
		counter = 0;
	}

	/*
	int i;
	struct onvm_pkt_meta* pmeta;
	for (i = 0; i < pkt_num; i ++) {
		pmeta = onvm_get_pkt_meta(recv_pkts[i]);
		pmeta->chain_index = 0;
		pmeta->action = ONVM_NF_ACTION_TONF;
	}
	*/

	onvm_nflib_send_processed(recv_pkts, pkt_num, 0);

	return 0;
}
#endif

static void
generate_pkts(void)
{
	struct rte_mempool *pktmbuf_pool;
	int i, mark;
	static uint32_t counter = 0;


	uint32_t gen_ip = 0;
	uint16_t gen_port = 0;

	pktmbuf_pool = rte_mempool_lookup(PKTMBUF_POOL_NAME);
	if (pktmbuf_pool == NULL) {
		onvm_nflib_stop();
		rte_exit(EXIT_FAILURE, "Cannot find mbuf pool!\n");
	}

	struct ether_hdr *ethh;
	struct ipv4_hdr *iph;
	struct udp_hdr *udph;

LOOP:
	for (i = 0; i < NUM_PKTS; i ++) {
		struct onvm_pkt_meta* pmeta;
		mark = 0;
		do {
			pkts[i] = rte_pktmbuf_alloc(pktmbuf_pool);
			if ((pkts[i] == NULL) && (mark == 0)) {
				RTE_LOG(DEBUG, APP, "Pktgen speed too fast, pktmbuf pool is exausted\n");
				mark = 1;
			}
		} while (pkts[i] == NULL); 
		pmeta = onvm_get_pkt_meta(pkts[i]);
		/* I am the first one in the service chain */
		pmeta->chain_index = 0;
		pmeta->action = ONVM_NF_ACTION_TONF;

		pkts[i]->port = 3;
		pkts[i]->hash.rss = i;

		ethh = onvm_pkt_ether_hdr(pkts[i]);
		//ethh->s_addr = PKTGEN_MAC_ADDR;
		ethh->ether_type = rte_cpu_to_be_16((uint16_t)(ETHER_TYPE_IPv4));

		iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
		iph->version_ihl = 0x40 | 0x05;
		iph->type_of_service = 0;
		/* total_length - 20 bytes ip header length */
		iph->total_length = rte_cpu_to_be_16((uint16_t)PKTGEN_IP_PKT_LENGTH);
		iph->packet_id = 0;
		iph->fragment_offset = 0;
		iph->time_to_live = 64;
		iph->next_proto_id = IPPROTO_UDP;
		iph->hdr_checksum = 0;
		iph->src_addr = gen_ip ++;

		udph = (struct udp_hdr *)((unsigned char *)iph + sizeof(struct ipv4_hdr));
		udph->src_port = rte_cpu_to_be_16((uint16_t)gen_port++);
		udph->dst_port = rte_cpu_to_be_16((uint16_t)PKTGEN_UDP_PORT);
		/* 20 bytes ip header and 8 bytes udp header */
		udph->dgram_len = rte_cpu_to_be_16((uint16_t)PKTGEN_IP_PKT_LENGTH - 20); 
		udph->dgram_cksum = 0;
	}

	/* Send it */
	onvm_nflib_send_processed(pkts, NUM_PKTS, 0);

	counter += NUM_PKTS;
	if (counter >= print_delay) {
		do_stats_display();
		counter = 0;
	}

	goto LOOP;
}

int
main(int argc, char *argv[]) {
	int arg_offset;

	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_PKTGEN, NULL)) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	generate_pkts();

#if !defined(KEEP_SENDING)
	onvm_nflib_run(&(loop_forward), 0);
#endif

	printf("If we reach here, program is ending\n");
	return 0;
}
