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

                                  onvm_init.c

                  File containing initialization functions.


******************************************************************************/

// CLEAR: 1

#include "onvm_init.h"
#include "manager.h"


/********************************Global variables*****************************/


struct client *clients = NULL;
struct port_info *ports = NULL;

struct rte_mempool *pktmbuf_pool;
struct rte_mempool *nf_info_pool;
pstack_thread_info pstack_info;

struct rte_ring *nf_info_queue;

uint16_t **services;
uint16_t *nf_per_service_count;
struct onvm_service_chain *default_chain;
struct onvm_service_chain **default_sc_p;


/*************************Internal Functions Prototypes***********************/


static int init_mbuf_pools(void);
static int init_client_info_pool(void);
static int init_port(uint8_t port_num);
static int init_shm_rings(void);
static int init_info_queue(void);
static void check_all_ports_link_status(uint8_t port_num, uint32_t port_mask);
static int init_pstack_info_pool(void);

/*********************************Interfaces**********************************/


int
init(int argc, char *argv[]) {
	int retval;
	const struct rte_memzone *mz_scp;
	uint8_t i, total_ports;

	/* init EAL, parsing EAL args */
	retval = rte_eal_init(argc, argv);
	if (retval < 0)
		return -1;
	argc -= retval;
	argv += retval;

	/* get total number of ports */
	total_ports = rte_eth_dev_count();

	/* set up ports info */
	ports = rte_malloc(MZ_PORT_INFO, sizeof(*ports), 0);
	if (ports == NULL)
		rte_exit(EXIT_FAILURE, "Cannot allocate memory for ports details\n");

	/* parse additional, application arguments */
	retval = parse_app_args(total_ports, argc, argv);
	if (retval != 0)
		return -1;

	/* initialise mbuf pools */
	retval = init_mbuf_pools();
	if (retval != 0)
		rte_exit(EXIT_FAILURE, "Cannot create needed mbuf pools\n");

	/* initialise client info pool */
	retval = init_client_info_pool();
	if (retval != 0) {
		rte_exit(EXIT_FAILURE, "Cannot create client info mbuf pool: %s\n", rte_strerror(rte_errno));
	}

	/* now initialise the ports we will use */
	for (i = 0; i < ports->num_ports; i++) {
		retval = init_port(ports->id[i]);
		if (retval != 0)
			rte_exit(EXIT_FAILURE, "Cannot initialise port %u\n",
					(unsigned)i);
	}

	check_all_ports_link_status(ports->num_ports, (~0x0));

	/* initialise the client queues/rings for inter-eu comms */
	init_shm_rings();

	/* initialise a queue for newly created NFs */
	init_info_queue();

	/* initialize gpu related settings */
	init_manager();

	init_pstack_info_pool();

	/* Choose service chain, copy one and paste out of "if 0" to use it */
	const int service_chain[MAX_SERVICES] = {NF_IPSEC, NF_FIREWALL, NF_NIDS, NF_ROUTER, NF_END};

#if 0
	/* 1 NF */
	const int service_chain[MAX_SERVICES] = {NF_ROUTER, NF_END};
	const int service_chain[MAX_SERVICES] = {NF_FIREWALL, NF_END};
	const int service_chain[MAX_SERVICES] = {NF_IPSEC, NF_END};
	const int service_chain[MAX_SERVICES] = {NF_NIDS, NF_END};
	/* 2 NFs */
	const int service_chain[MAX_SERVICES] = {NF_ROUTER, NF_FIREWALL, NF_END};
	const int service_chain[MAX_SERVICES] = {NF_FIREWALL, NF_NIDS, NF_END};
	const int service_chain[MAX_SERVICES] = {NF_IPSEC, NF_NIDS, NF_END};
	/* 3 NFs */
	const int service_chain[MAX_SERVICES] = {NF_FIREWALL, NF_NIDS, NF_IPSEC, NF_END};
	const int service_chain[MAX_SERVICES] = {NF_NIDS, NF_IPSEC, NF_ROUTER, NF_END};
	/* 4 NFs */
	const int service_chain[MAX_SERVICES] = {NF_FIREWALL, NF_NIDS, NF_IPSEC, NF_ROUTER, NF_END};
#endif


	/* initialize a default service chain*/
	default_chain = onvm_sc_create();

	/* Construct the service chain  */
	for (i = 0; i < MAX_SERVICES; i ++) {
		if (service_chain[i] == NF_END) {
			break;
		}

		retval = onvm_sc_append_entry(default_chain, ONVM_NF_ACTION_TONF, service_chain[i]);
		if (retval == ENOSPC)
			rte_exit(EXIT_FAILURE, "chain length can not be larger than the maximum chain length\n");
	}

	/* Add the last action */
	retval = onvm_sc_append_entry(default_chain, ONVM_NF_ACTION_OUT, 0);
	if (retval == ENOSPC)
		rte_exit(EXIT_FAILURE, "chain length can not be larger than the maximum chain length\n");

	/* set up service chain pointer shared to NFs*/
	mz_scp = rte_memzone_reserve(MZ_SCP_INFO, sizeof(struct onvm_service_chain *),
			rte_socket_id(), NO_FLAGS);
	if (mz_scp == NULL)
		rte_exit(EXIT_FAILURE, "Canot reserve memory zone for service chain pointer\n");
	memset(mz_scp->addr, 0, sizeof(struct onvm_service_chain *));
	default_sc_p = mz_scp->addr;
	*default_sc_p = default_chain;
	onvm_sc_print(default_chain);

	return 0;
}


/*****************************Internal functions******************************/


/**
 * Initialise the mbuf pool for packet reception for the NIC, and any other
 * buffer pools needed by the app - currently none.
 */
static int
init_mbuf_pools(void) {
	const unsigned num_mbufs = (MAX_CLIENTS * MBUFS_PER_CLIENT) + (ports->num_ports * MBUFS_PER_PORT);

	/* don't pass single-producer/single-consumer flags to mbuf create as it
	 * seems faster to use a cache instead */
	printf("Creating mbuf pool '%s' [%u mbufs] ...\n", PKTMBUF_POOL_NAME, num_mbufs);

	pktmbuf_pool = rte_mempool_create(PKTMBUF_POOL_NAME, num_mbufs,
			MBUF_SIZE, MBUF_CACHE_SIZE,
			sizeof(struct rte_pktmbuf_pool_private), rte_pktmbuf_pool_init,
			NULL, rte_pktmbuf_init, NULL, rte_socket_id(), NO_FLAGS);

	return (pktmbuf_pool == NULL); /* 0  on success */
}

/**
 * Set up a mempool to store nf_info structs
 */
static int
init_client_info_pool(void) {
	/* don't pass single-producer/single-consumer flags to mbuf
	 * create as it seems faster to use a cache instead */
	printf("Creating mbuf pool '%s' ...\n", _NF_MEMPOOL_NAME);
	nf_info_pool = rte_mempool_create(_NF_MEMPOOL_NAME, MAX_CLIENTS,
			NF_INFO_SIZE, NF_INFO_CACHE,
			0, NULL, NULL, NULL, NULL, rte_socket_id(), NO_FLAGS);

	return (nf_info_pool == NULL); /* 0 on success */
}

/**
 * Set up the mempool of the common tcp/ip stack (pstack)
 */

static int
init_pstack_info_pool(void) {
	// pstack_info.ip_thread_local = rte_mempool_create(MZ_PSTACK_IP_INFO_NAME, ONVM_NUM_RX_THREADS,
	// 		MZ_PSTACK_IP_INFO_SIZE, NF_INFO_CACHE, // TODO cache size?
	// 		0, NULL, NULL, NULL, NULL, rte_socket_id(), NO_FLAGS);
	// pstack_info.tcp_thread_local = rte_mempool_create(MZ_PSTACK_TCP_INFO_NAME, ONVM_NUM_RX_THREADS,
	// 		MZ_PSTACK_TCP_INFO_SIZE, NF_INFO_CACHE, // TODO cache size?
	// 		0, NULL, NULL, NULL, NULL, rte_socket_id(), NO_FLAGS);

	// USE RTE_MALLOC for now 
	// The pstack library use dynamic memory allocation
	// shared memory requires rte_mempool
	pstack_info.ip_thread_local = rte_malloc(PSTACK_IP_INFO_NAME, ONVM_NUM_RX_THREADS * PSTACK_IP_INFO_SIZE, 0);
	pstack_info.tcp_thread_local = rte_malloc(PSTACK_TCP_INFO_NAME, ONVM_NUM_RX_THREADS * PSTACK_TCP_INFO_SIZE, 0);

	pstack_init(pstack_info, ONVM_NUM_RX_THREADS);

	return (pstack_info.ip_thread_local == NULL || pstack_info.tcp_thread_local == NULL); 
}

/**
 * Initialise an individual port:
 * - configure number of rx and tx rings
 * - set up each rx ring, to pull from the main mbuf pool
 * - set up each tx ring
 * - start the port and report its status to stdout
 */
static int
init_port(uint8_t port_num) {

	static uint8_t rss_symmetric_key[40] = { 0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,
											0x6d, 0x5a, 0x6d, 0x5a,};

	/* for port configuration all features are off by default */
	const struct rte_eth_conf port_conf = {
		.rxmode = {
			.mq_mode = ETH_MQ_RX_RSS
		},
		.rx_adv_conf = {
			.rss_conf = {
				.rss_key = rss_symmetric_key,
				.rss_hf = ETH_RSS_IP | ETH_RSS_UDP | ETH_RSS_TCP,
			}
		},
	};

	const uint16_t rx_rings = ONVM_NUM_RX_THREADS, tx_rings = ONVM_NUM_TX_THREADS_PER_PORT;
	const uint16_t rx_ring_size = RTE_MP_RX_DESC_DEFAULT;
	const uint16_t tx_ring_size = RTE_MP_TX_DESC_DEFAULT;

	uint16_t q;
	int retval;

	printf("Port %u init ... \n", (unsigned)port_num);
	printf("Port %u socket id %u ... \n", (unsigned)port_num, (unsigned)rte_eth_dev_socket_id(port_num));
	printf("Port %u Rx rings %u ... \n", (unsigned)port_num, (unsigned)rx_rings);
	fflush(stdout);

	/* Standard DPDK port initialisation - config port, then set up
	 * rx and tx rings */
	if ((retval = rte_eth_dev_configure(port_num, rx_rings, tx_rings,
					&port_conf)) != 0)
		return retval;

	for (q = 0; q < rx_rings; q++) {
		retval = rte_eth_rx_queue_setup(port_num, q, rx_ring_size,
				rte_eth_dev_socket_id(port_num),
				NULL, pktmbuf_pool);
		if (retval < 0) return retval;
	}

	for (q = 0; q < tx_rings; q++) {
		retval = rte_eth_tx_queue_setup(port_num, q, tx_ring_size,
				rte_eth_dev_socket_id(port_num),
				NULL);
		if (retval < 0) return retval;
	}

	rte_eth_promiscuous_enable(port_num);

	retval = rte_eth_dev_start(port_num);
	if (retval < 0) return retval;

	// ports->tx_q_new[port_num] = rte_ring_create(get_port_tx_queue_name(port_num),
	// 		BATCH_QUEUE_FACTOR * MAX_BATCH_SIZE, rte_socket_id(), NO_FLAGS);
	// if (!ports->tx_q_new[port_num])
	// 	return -1;

	for (q = 0; q < ONVM_NUM_NF_QUEUES; q++) {
		printf("Doing: %d\n", q);
		ports->tx_qs[port_num][q] = rte_ring_create(get_port_tx_multi_queue_name(port_num, q),
			BATCH_QUEUE_FACTOR * MAX_BATCH_SIZE, rte_socket_id(), RING_F_SP_ENQ);
		if (!ports->tx_qs[port_num][q])
			return -1;
	}

	printf("done: \n");

	return 0;
}

/**
 * Set up the DPDK rings which will be used to pass packets, via
 * pointers, between the multi-process server and client processes.
 * Each client needs one RX queue.
 */
static int
init_shm_rings(void) {
	unsigned i, j;
	unsigned socket_id;
	const struct rte_memzone *mz;

	// use calloc since we allocate for all possible clients
	// ensure that all fields are init to 0 to avoid reading garbage
	// TODO plopreiato, move to creation when a NF starts
	mz = rte_memzone_reserve(MZ_CLIENTS, MAX_CLIENTS * sizeof(*clients), rte_socket_id(), NO_FLAGS);
	if (!mz || !mz->addr)
		rte_exit(EXIT_FAILURE, "Cannot allocate memory for client program details\n");
	clients = mz->addr;
	memset(clients, 0, MAX_CLIENTS * sizeof(*clients));

	services = rte_calloc("service to nf map", num_services, sizeof(uint16_t*), 0);
	for (i = 0; i < num_services; i++) {
		services[i] = rte_calloc("one service NFs", MAX_CLIENTS_PER_SERVICE, sizeof(uint16_t), 0);
	}
	nf_per_service_count = rte_calloc("count of NFs active per service", num_services, sizeof(uint16_t), 0);
	if (services == NULL || nf_per_service_count == NULL)
		rte_exit(EXIT_FAILURE, "Cannot allocate memory for service to NF mapping\n");

	for (i = 0; i < MAX_CLIENTS; i++) {
		/* Create an RX queue for each client */
		socket_id = rte_socket_id();
		clients[i].instance_id = i;

		// clients[i].rx_q_new = rte_ring_create(get_rx_queue_name(i, 0), BATCH_QUEUE_FACTOR * MAX_BATCH_SIZE, socket_id, NO_FLAGS);
		clients[i].tx_q_new = NULL;
		// UNUSED(j);
		/* rss ring for NF */
		for (j = 0; j < ONVM_NUM_NF_QUEUES; j++) {
			clients[i].rx_qs[j] = rte_ring_create(get_rx_queue_name(i, j), 
						BATCH_QUEUE_FACTOR * MAX_BATCH_SIZE / ONVM_NUM_NF_QUEUES, socket_id, RING_F_SP_ENQ | RING_F_SC_DEQ); // single producer & consumer
			clients[i].tx_qs[j] = NULL;
		}

		rte_spinlock_init(&clients[i].stats.update_lock);
	}
	return 0;
}

/**
 * Allocate a rte_ring for newly created NFs
 */
static int
init_info_queue(void) {
	nf_info_queue = rte_ring_create(
			_NF_QUEUE_NAME,
			MAX_CLIENTS,
			rte_socket_id(),
			RING_F_SC_DEQ); // MP enqueue (default), SC dequeue

	if (nf_info_queue == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create nf info queue\n");

	return 0;
}

/* Check the link status of all ports in up to 9s, and print them finally */
static void
check_all_ports_link_status(uint8_t port_num, uint32_t port_mask) {
#define CHECK_INTERVAL 100 /* 100ms */
#define MAX_CHECK_TIME 90 /* 9s (90 * 100ms) in total */
	uint8_t portid, count, all_ports_up, print_flag = 0;
	struct rte_eth_link link;

	printf("\nChecking link status");
	fflush(stdout);
	for (count = 0; count <= MAX_CHECK_TIME; count++) {
		all_ports_up = 1;
		for (portid = 0; portid < port_num; portid++) {
			if ((port_mask & (1 << ports->id[portid])) == 0)
				continue;
			memset(&link, 0, sizeof(link));
			rte_eth_link_get_nowait(ports->id[portid], &link);
			/* print link status if flag set */
			if (print_flag == 1) {
				if (link.link_status)
					printf("Port %d Link Up - speed %u "
							"Mbps - %s\n", ports->id[portid],
							(unsigned)link.link_speed,
							(link.link_duplex == ETH_LINK_FULL_DUPLEX) ?
							("full-duplex") : ("half-duplex\n"));
				else
					printf("Port %d Link Down\n",
							(uint8_t)ports->id[portid]);
				continue;
			}
			/* clear all_ports_up flag if any link down */
			if (link.link_status == 0) {
				all_ports_up = 0;
				break;
			}
		}
		/* after finally printing all link status, get out */
		if (print_flag == 1)
			break;

		if (all_ports_up == 0) {
			printf(".");
			fflush(stdout);
			rte_delay_ms(CHECK_INTERVAL);
		}

		/* set the print_flag if all ports up or timeout */
		if (all_ports_up == 1 || count == (MAX_CHECK_TIME - 1)) {
			print_flag = 1;
			printf("done\n");
		}
	}
}

/**
 * Main init function for the multi-process server app,
 * calls subfunctions to do each stage of the initialisation.
 */
