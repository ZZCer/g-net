/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2016 George Washington University
 *            2015-2016 University of California Riverside
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

                                  onvm_nflib.c


                  File containing all functions of the NF API


******************************************************************************/


/***************************Standard C library********************************/


#include <getopt.h>
#include <signal.h>


/*****************************Internal headers********************************/


#include "onvm_nflib.h"
#include "onvm_includes.h"
#include "onvm_sc_common.h"
#include "onvm_framework.h"
#include "fifo.h"

/******************************Global Variables*******************************/

// ring used to place new nf_info struct
static struct rte_ring *nf_info_ring;


#if !defined(BQUEUE_SWITCH)
// rings used to pass packets between NFlib and NFmgr
static struct rte_ring *tx_ring, *rx_ring[MAX_CPU_THREAD_NUM];
#endif


// shared data from server. We update statistics here
volatile struct client_tx_stats *tx_stats;


// Shared data for client info
struct onvm_nf_info *nf_info;


// Shared pool for all clients info
static struct rte_mempool *nf_info_mp;


// User-given NF Client ID (defaults to manager assigned)
static uint16_t initial_instance_id = NF_NO_ID;


// True as long as the NF should keep processing packets
volatile uint8_t keep_running = 1;


// Shared data for default service chain
static struct onvm_service_chain *default_chain;

// request and response memory pool
struct rte_mempool *nf_request_mp;
struct rte_mempool *nf_response_mp;

// request and response queue
struct rte_ring *nf_request_queue;

extern int NF_REQUIRED_LATENCY;
extern int INIT_WORKER_THREAD_NUM;

/***********************Internal Functions Prototypes*************************/


/*
 * Function that initialize a nf info data structure.
 *
 * Input  : the tag to name the NF
 * Output : the data structure initialized
 *
 */
static struct onvm_nf_info *
onvm_nflib_info_init(const char *tag, int service_id);


/*
 * Function printing an explanation of command line instruction for a NF.
 *
 * Input : name of the executable containing the NF
 *
 */
static void
onvm_nflib_usage(const char *progname);


/*
 * Function that parses the global arguments common to all NFs.
 *
 * Input  : the number of arguments (following C standard library convention)
 *          an array of strings representing these arguments
 * Output : an error code
 *
 */
static int
onvm_nflib_parse_args(int argc, char *argv[]);


/*
* Signal handler to catch SIGINT.
*
* Input : int corresponding to the signal catched
*
*/
void
onvm_nflib_handle_signal(int sig);

/*
 * Set this NF's status to not running and release memory
 *
 * Input: Info struct corresponding to this NF
 */
static void
onvm_nflib_cleanup(void);

/************************************API**************************************/


int
onvm_nflib_init(int argc, char *argv[], const char *nf_tag, int service_id,
		void (*user_install_gpu_rule)(void)) {
	const struct rte_memzone *mz;
	const struct rte_memzone *mz_scp;
	struct rte_mempool *mp;
	struct onvm_service_chain **scp;
	int retval_eal, retval_parse, retval_final;

	if (service_id < 0)
		rte_exit(EXIT_FAILURE, "Service ID not set");

	if ((retval_eal = rte_eal_init(argc, argv)) < 0)
		return -1;

	/* Modify argc and argv to conform to getopt rules for parse_nflib_args */
	argc -= retval_eal; argv += retval_eal;

	/* Reset getopt global variables opterr and optind to their default values */
	opterr = 0; optind = 1;

	if ((retval_parse = onvm_nflib_parse_args(argc, argv)) < 0)
		rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");

	/*
	 * Calculate the offset that the nf will use to modify argc and argv for its
	 * getopt call. This is the sum of the number of arguments parsed by
	 * rte_eal_init and parse_nflib_args. This will be decremented by 1 to assure
	 * getopt is looking at the correct index since optind is incremented by 1 each
	 * time "--" is parsed.
	 * This is the value that will be returned if initialization succeeds.
	 */
	retval_final = (retval_eal + retval_parse) - 1;

	/* Reset getopt global variables opterr and optind to their default values */
	opterr = 0; optind = 1;

	/* Lookup mempool for nf_info struct */
	nf_info_mp = rte_mempool_lookup(_NF_MEMPOOL_NAME);
	if (nf_info_mp == NULL)
		rte_exit(EXIT_FAILURE, "No Client Info mempool - bye\n");

	/* Initialize the info struct */
	nf_info = onvm_nflib_info_init(nf_tag, service_id);

	mp = rte_mempool_lookup(PKTMBUF_POOL_NAME);
	if (mp == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get mempool for mbufs\n");

	mz = rte_memzone_lookup(MZ_CLIENT_INFO);
	if (mz == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get tx info structure\n");
	tx_stats = mz->addr;

	mz_scp = rte_memzone_lookup(MZ_SCP_INFO);
	if (mz_scp == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get service chain info structre\n");
	scp = mz_scp->addr;
	default_chain = *scp;

	onvm_sc_print(default_chain);

	nf_info_ring = rte_ring_lookup(_NF_QUEUE_NAME);
	if (nf_info_ring == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get nf_info ring");

	/* Put this NF's info struct onto queue for manager to process startup */
	if (rte_ring_enqueue(nf_info_ring, nf_info) < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager");
	}

	/* Wait for a client id to be assigned by the manager */
	RTE_LOG(INFO, APP, "Waiting for manager to assign an ID...\n");
	for (; nf_info->status == (uint16_t)NF_WAITING_FOR_ID;) {
		sleep(1);
	}

	/* This NF is trying to declare an ID already in use. */
	if (nf_info->status == NF_ID_CONFLICT) {
		rte_mempool_put(nf_info_mp, nf_info);
		rte_exit(NF_ID_CONFLICT, "Selected ID already in use. Exiting...\n");
	} else if (nf_info->status == NF_NO_IDS) {
		rte_mempool_put(nf_info_mp, nf_info);
		rte_exit(NF_NO_IDS, "There are no ids available for this NF\n");
	} else if (nf_info->status != NF_STARTING) {
		rte_mempool_put(nf_info_mp, nf_info);
		rte_exit(EXIT_FAILURE, "Error occurred during manager initialization\n");
	}
	RTE_LOG(INFO, APP, "Using Instance ID %d\n", nf_info->instance_id);
	RTE_LOG(INFO, APP, "Using Service ID %d\n", nf_info->service_id);

	/* Install GPU rules, including required latency and so on */
	if ((service_id != NF_PKTGEN) && (service_id != NF_RAW)) {
		assert(user_install_gpu_rule != NULL);
		user_install_gpu_rule();
	}

#if !defined(BQUEUE_SWITCH)
	/* Now, map rx and tx rings into client space */
	unsigned i;
	for (i = 0; i < MAX_CPU_THREAD_NUM; i ++) {
		rx_ring[i] = rte_ring_lookup(get_rx_queue_name(nf_info->instance_id, i));
		if (rx_ring[i] == NULL)
			rte_exit(EXIT_FAILURE, "Cannot get RX ring - is server process running?\n");
	}

	tx_ring = rte_ring_lookup(get_tx_queue_name(nf_info->instance_id));
	if (tx_ring == NULL)
		rte_exit(EXIT_FAILURE, "Cannot get TX ring - is server process running?\n");
#endif

	/* Tell the manager we're ready to recieve packets */
	nf_info->status = NF_RUNNING;

	/* Get request memory pool */
	nf_request_mp = rte_mempool_lookup(_NF_REQUEST_MEMPOOL_NAME);
	if (nf_request_mp == NULL)
		rte_exit(EXIT_FAILURE, "Failed to get request mempool\n");

	/* Get response memory pool */
	nf_response_mp = rte_mempool_lookup(_NF_RESPONSE_MEMPOOL_NAME);
	if (nf_response_mp == NULL)
		rte_exit(EXIT_FAILURE, "Failed to get response mempool\n");

	/* Get request ring */
	nf_request_queue = rte_ring_lookup(_NF_REQUEST_QUEUE_NAME);
	if (nf_request_queue == NULL)
		rte_exit(EXIT_FAILURE, "Failed to get request ring\n");

	RTE_LOG(INFO, APP, "Finished Process Init.\n");
	return retval_final;
}


#if defined(BQUEUE_SWITCH)
int
onvm_nflib_run(int(*handler)(int thread_id), int thread_id)
{
	RTE_LOG(DEBUG, APP, "\nClient process %d, thread %d handling packets\n", nf_info->instance_id, thread_id);

	/* Listen for ^C and docker stop so we can exit gracefully */
	signal(SIGINT, onvm_nflib_handle_signal);
	signal(SIGTERM, onvm_nflib_handle_signal);

	/* loop inside */
	(*handler)(thread_id);

	return 0;
}
#else
int
onvm_nflib_run(int(*handler)(struct rte_mbuf **pkt, int nb_pkt, int thread_id), int thread_id)
{
	RTE_LOG(DEBUG, APP, "\nClient process %d, thread %d handling packets\n", nf_info->instance_id, thread_id);

	/* Listen for ^C and docker stop so we can exit gracefully */
	signal(SIGINT, onvm_nflib_handle_signal);
	signal(SIGTERM, onvm_nflib_handle_signal);

	void *pkts[PKT_READ_SIZE];
	uint16_t nb_pkts;

	for (; keep_running;) {
		/* Dequeue all packets in ring up to max possible. */
		/* TODO: shall each cpu thread has a rx ring? How to load balance? */
		nb_pkts = rte_ring_dequeue_burst(rx_ring[thread_id], pkts, PKT_READ_SIZE);
		if (nb_pkts != 0)
			RTE_LOG(DEBUG, APP, "[%d] receives %d packets\n", thread_id, nb_pkts);

#if defined(NF_RX_SPEED_TEST)
		int i = 0;
		for (i = 0; i < nb_pkts; i ++) {
			rte_pktmbuf_free(pkts[i]);
		}
		continue;
#endif

		/* Give each packet to the user processing function */
		(*handler)((struct rte_mbuf **)pkts, nb_pkts, thread_id);
	}

	// Stop and free
	onvm_nflib_cleanup();


	return 0;
}

int
onvm_nflib_send_processed(struct rte_mbuf **pkt_ptr_buf, int tx_batch_size, int thread_id)
{
	int tx_go;
	int progress = 0;
	int free_slots;

enqueue_again:
	/* Avoid the batch size getting too large that can never be sent out.
	 * The rte_ring_free_count() should not be placed in MIN() macro, because when it is
	 * calculated twice in the macro, the first one and the second one can be different,
	 * leading to a bug */
	free_slots = rte_ring_free_count(tx_ring);
	tx_go = GENERIC_MIN(free_slots, (uint16_t)(tx_batch_size - progress));

	if (unlikely((tx_go > 0) && (rte_ring_enqueue_bulk(tx_ring, (void * const*)&(pkt_ptr_buf[progress]), tx_go, NULL) == -ENOBUFS))) {
		/* Threads may compete to enqueue, therefore the free_slots can be changed by others. */
		RTE_LOG(DEBUG, APP, "[%d] %u available free entries in the tx_ring, batch size %d\n",
				thread_id, rte_ring_free_count(tx_ring), tx_batch_size);
		goto enqueue_again;
	} else {
		if (thread_id == 0)
			tx_stats[nf_info->instance_id].tx += tx_go;
		progress += tx_go;
		RTE_LOG(DEBUG, APP, "[%d] Send %d packets\n", thread_id, tx_go);
	}

	if (tx_batch_size - progress > 0)
		goto enqueue_again;

	return 0;
}
#endif /* BQUEUE_SWITCH */

void
onvm_nflib_stop(void) {
	onvm_nflib_cleanup();
}

/******************************Helper functions*******************************/


static struct onvm_nf_info *
onvm_nflib_info_init(const char *tag, int service_id)
{
	void *mempool_data;
	struct onvm_nf_info *info;

	if (rte_mempool_get(nf_info_mp, &mempool_data) < 0)
		rte_exit(EXIT_FAILURE, "Failed to get client info memory");

	if (mempool_data == NULL)
		rte_exit(EXIT_FAILURE, "Client Info struct not allocated");

	info = (struct onvm_nf_info*) mempool_data;
	info->instance_id = initial_instance_id;
	info->service_id = service_id;
	info->status = NF_WAITING_FOR_ID;
	info->tag = tag;

	return info;
}


static void
onvm_nflib_usage(const char *progname) {
	printf("Usage: %s [EAL args] -- "
			"[-n <instance_id>]"
			"[-r <service_id>]\n\n", progname);
}


static int
onvm_nflib_parse_args(int argc, char *argv[]) {
	const char *progname = argv[0];
	int c;

	opterr = 0;
	while ((c = getopt (argc, argv, "n:l:k:")) != -1)
		switch (c) {
			case 'n':
				initial_instance_id = (uint16_t) strtoul(optarg, NULL, 10);
				RTE_LOG(INFO, APP, "[ARG] Initial_instance_id = %d\n", initial_instance_id);
				break;
			case 'l':
				NF_REQUIRED_LATENCY = (uint16_t) strtoul(optarg, NULL, 10);
				RTE_LOG(INFO, APP, "[ARG] NF required latency = %d microseconds (us)\n", NF_REQUIRED_LATENCY);
				break;
			case 'k':
				INIT_WORKER_THREAD_NUM = (uint16_t) strtoul(optarg, NULL, 10);
				RTE_LOG(INFO, APP, "[ARG] Initial worker thread number = %d\n", INIT_WORKER_THREAD_NUM);
				break;
			case '?':
				onvm_nflib_usage(progname);
				if (optopt == 'n')
					fprintf(stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint(optopt))
					fprintf(stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
				return -1;
			default:
				return -1;
		}

	return optind;
}


void
onvm_nflib_handle_signal(int sig)
{
	if (sig == SIGINT || sig == SIGTERM)
		keep_running = 0;
	printf("signal catched\n");
}

static void
onvm_nflib_cleanup(void)
{
	nf_info->status = NF_STOPPED;

	/* Put this NF's info struct back into queue for manager to ack shutdown */
	nf_info_ring = rte_ring_lookup(_NF_QUEUE_NAME);
	if (nf_info_ring == NULL) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot get nf_info ring for shutdown");
	}

	if (rte_ring_enqueue(nf_info_ring, nf_info) < 0) {
		rte_mempool_put(nf_info_mp, nf_info); // give back mermory
		rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for shutdown");
	}

}
