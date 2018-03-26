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
#include <rte_cycles.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#define NF_TAG "raw_nf"

/* number of package between each print */
static uint32_t print_delay = 50000;

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
	printf("TX pkts per second: %9"PRIu64" \n", (cur_pkts - last_pkts)
			* rte_get_timer_hz() / (cur_cycles - last_cycles));

	last_pkts = cur_pkts;
	last_cycles = cur_cycles;

	printf("\n\n");
}

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

	onvm_nflib_send_processed(recv_pkts, pkt_num, 0);

	return 0;
}

int
main(int argc, char *argv[]) {
	int arg_offset;

	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_RAW, NULL)) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	onvm_nflib_run(&(loop_forward), 0);

	printf("If we reach here, program is ending\n");
	return 0;
}
