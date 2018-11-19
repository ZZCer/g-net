#define _GNU_SOURCE
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>
#include <pthread.h>

#include <sys/queue.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>

#include <rte_common.h>
#include <rte_errno.h>
#include <rte_byteorder.h>
#include <rte_log.h>
#include <rte_debug.h>
#include <rte_cycles.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_atomic.h>
#include <rte_branch_prediction.h>
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_interrupts.h>
#include <rte_pci.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_dev.h>
#include <rte_string_fns.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_random.h>
#include <sys/stat.h>

//We assume these values are host value (little-endian)
#define IPV4_HDR_DF_SHIFT           14
#define IPV4_HDR_MF_SHIFT           13
#define IPV4_HDR_FO_SHIFT           3

#define IPV4_HDR_DF_MASK            (1 << IPV4_HDR_DF_SHIFT)
#define IPV4_HDR_MF_MASK            (1 << IPV4_HDR_MF_SHIFT)
#define IPV4_HDR_FO_MASK            ((1 << IPV4_HDR_FO_SHIFT) - 1)
#define PORT_NUM 1
#define MEMPOOL_NUM 1
#define MBUF_NUM 32

#define RX_RING_SIZE 128
#define TX_RING_SIZE 512

#define LOCAL_IP_ADDR (uint32_t)(456)
#define KV_IP_ADDR (uint32_t)(789)
#define LOCAL_UDP_PORT (uint16_t)(123)
#define KV_UDP_PORT (uint16_t)(124)

#define NUM_TX_QUEUE 4
#define NUM_RX_QUEUE 1

// #define INCREASE_TTL

int PKTLEN;

#ifdef INCREASE_TTL
uint8_t ttl = 0;
#else
uint8_t ttl = 64;
#endif
struct rte_mempool *mp;

/*
 *  * Ethernet device configuration.
 *   */
static struct rte_eth_rxmode rx_mode = {
	.max_rx_pkt_len = ETHER_MAX_LEN, /**< Default maximum frame length. */
	.split_hdr_size = 0, 
	.header_split   = 0, /**< Header Split disabled. */
	.hw_ip_checksum = 0, /**< IP checksum offload disabled. */
	.hw_vlan_filter = 0, /**< VLAN filtering disabled. */
	.hw_vlan_strip  = 0, /**< VLAN strip disabled. */
	.hw_vlan_extend = 0, /**< Extended VLAN disabled. */
	.jumbo_frame    = 0, /**< Jumbo Frame Support disabled. */
	.hw_strip_crc   = 0, /**< CRC stripping by hardware disabled. */
};

static struct rte_eth_txmode tx_mode = {
	.mq_mode = ETH_MQ_TX_NONE
};

#define NUM_MAX_CORE 6
struct benchmark_core_statistics {
	uint64_t tx;
	uint64_t rx;
	uint64_t dropped;
	int enable;
} __rte_cache_aligned;
struct benchmark_core_statistics core_statistics[NUM_MAX_CORE];

typedef struct context_s {
	unsigned int core_id;
	unsigned int queue_id;
	unsigned int port_id;
} context_t;


/* A tsc-based timer responsible for triggering statistics printout */
#define TIMER_MILLISECOND 2000000ULL /* around 1ms at 2 Ghz */
#define MAX_TIMER_PERIOD 86400 /* 1 day max */
static int64_t timer_period = 3 * TIMER_MILLISECOND * 1000; /* default period is 3 seconds */

struct timeval startime;
struct timeval endtime;
uint64_t ts_count[NUM_RX_QUEUE], ts_total[NUM_RX_QUEUE];

int loop_mark[NUM_MAX_CORE] = {0};

static struct rte_eth_conf port_conf_default;
static void
packet_ipv4hdr_constructor(struct ipv4_hdr *iph, int payload_len)
{
	iph->version_ihl = 0x40 | 0x05;
	iph->type_of_service = 0;
	iph->packet_id = 0;
	/* set DF flag */
	iph->fragment_offset = htons(IPV4_HDR_DF_MASK);
	iph->time_to_live = ttl;

	/* Total length of L3 */
	iph->total_length = htons(sizeof(struct ipv4_hdr) + sizeof(struct
				udp_hdr) + payload_len);

	iph->next_proto_id = IPPROTO_UDP;
	iph->src_addr = LOCAL_IP_ADDR;
	iph->dst_addr = KV_IP_ADDR;
}

#ifdef PRINT_INFO
static
void display_mac_address(struct ether_hdr *ethh, uint8_t port_id)
{
	printf("port_from %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
			" %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
			(unsigned)port_id,
			ethh->s_addr.addr_bytes[0], ethh->s_addr.addr_bytes[1],
			ethh->s_addr.addr_bytes[2], ethh->s_addr.addr_bytes[3],
			ethh->s_addr.addr_bytes[4], ethh->s_addr.addr_bytes[5]);
}
#endif

static void
packet_constructor_udp(char *pkt, uint8_t port_id, int payload_len)
{
	struct ether_hdr *ethh;
	struct ipv4_hdr *iph;
	struct udp_hdr *udph;
	char *data;
	uint16_t ip_ihl;

	ethh = (struct ether_hdr *)pkt;
	iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));

	/* 1. Ethernet headers for the packet */
	ethh->ether_type = htons(ETHER_TYPE_IPv4);
	rte_eth_macaddr_get(port_id, &(ethh->s_addr));
	
	/* 2. construct IP header */
	packet_ipv4hdr_constructor(iph, payload_len);

	/* 3. construct udp header */
	ip_ihl = (iph->version_ihl & 0x0f) * 4;
	assert(ip_ihl == sizeof(struct ipv4_hdr));
	udph = (struct udp_hdr *)((char *)iph + ip_ihl);

	udph->src_port = htons(LOCAL_UDP_PORT);
	udph->dst_port = htons(KV_UDP_PORT);
	udph->dgram_len = htons(8+payload_len);

	/* Init IPV4 and UDP checksum with 0 */
	iph->hdr_checksum = 0;
	udph->dgram_cksum = 0;

	/* calculate IPV4 and UDP checksum in SW */
	udph->dgram_cksum = rte_ipv4_udptcp_cksum(iph, udph);
	iph->hdr_checksum = rte_ipv4_cksum(iph);

	/* 4. payload */
	data = ((char *)udph + sizeof(struct udp_hdr));
	for(int i = 0; i < payload_len; i++) {
		*(data + i) = 1;
	}
}

static void
setup_mbuf(uint8_t port_id, struct rte_mempool *mp, struct rte_mbuf **tx_packets)
{
	char *pkt;
	int payload_len = PKTLEN - sizeof(struct ether_hdr) - sizeof(struct ipv4_hdr) - sizeof(struct udp_hdr);
	printf("Pkt len is %d, payload len is %d\n", PKTLEN, payload_len);

	for (int i = 0; i < MBUF_NUM; i++) {
		tx_packets[i] = rte_pktmbuf_alloc(mp);
		if (!tx_packets[i]) {
			printf("allocate mbuf failed\n");
			exit(1);
		}
		rte_pktmbuf_reset_headroom(tx_packets[i]);

		pkt = rte_pktmbuf_mtod(tx_packets[i], char *);
		packet_constructor_udp(pkt, port_id, payload_len);

		/*update mbuf metadata */
		//tx_packets[i]->pkt_len = sizeof(struct ipv4_hdr) + sizeof(struct udp_hdr) + payload_len;
		tx_packets[i]->pkt_len = PKTLEN;
		tx_packets[i]->data_len = tx_packets[i]->pkt_len;
		tx_packets[i]->nb_segs = 1;
		tx_packets[i]->ol_flags = 0;
		tx_packets[i]->l2_len = sizeof(struct ether_hdr);
		tx_packets[i]->l3_len = sizeof(struct ipv4_hdr);
	}
}

static void
init_mempool(struct rte_mempool **mempool)
{
	uint32_t nb_mbufs = MBUF_NUM * 100 * PORT_NUM * NUM_TX_QUEUE;
	uint16_t mbuf_size = RTE_MBUF_DEFAULT_BUF_SIZE;

	*mempool = rte_pktmbuf_pool_create("mempool0", nb_mbufs, 32, 0,
			mbuf_size, rte_socket_id());
}

/* Print out statistics on packets dropped */
static void
print_stats(void)
{
	uint64_t total_packets_dropped, total_packets_tx, total_packets_rx;
	uint64_t total_latency = 0, total_latency_cnt = 1;
	unsigned core_id, queue_id;

	total_packets_dropped = 0;
	total_packets_tx = 0;
	total_packets_rx = 0;

	const char clr[] = { 27, '[', '2', 'J', '\0' };
	const char topLeft[] = { 27, '[', '1', ';', '1', 'H','\0' };

	/* Clear screen and move to top left */
	printf("%s%s", clr, topLeft);

	static struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &end);
	double diff = 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000;

	printf("\n===== Core statistics ====================================");
	printf("\nNUM_TX_QUEUE = %d, NUM_RX_QUEUE = %d, PKT SIZE = %d", NUM_TX_QUEUE, NUM_RX_QUEUE, PKTLEN);

	for (core_id = 0; core_id < NUM_MAX_CORE; core_id ++) {
		if (core_statistics[core_id].enable == 0) continue;
		printf("\nStatistics for core %d ----------"
				"    Packets sent: %10"PRIu64
				";    Packets received: %10"PRIu64
				";    Packets dropped: %10"PRIu64,
				core_id,
				core_statistics[core_id].tx,
				core_statistics[core_id].rx,
				core_statistics[core_id].dropped);

		total_packets_dropped += core_statistics[core_id].dropped;
		total_packets_tx += core_statistics[core_id].tx;
		total_packets_rx += core_statistics[core_id].rx;

		core_statistics[core_id].dropped = 0;
		core_statistics[core_id].tx = 0;
		core_statistics[core_id].rx = 0;
	}

	for (queue_id = 0; queue_id < NUM_RX_QUEUE; queue_id ++) {
		total_latency += ts_total[queue_id];
		total_latency_cnt += ts_count[queue_id];
		ts_total[queue_id] = 0;
		ts_count[queue_id] = 0;
	}
	printf("\n===== Aggregate statistics ==============================="
			"\nTotal packets sent: %18"PRIu64
			"\nTotal packets received: %14"PRIu64
			"\nTotal packets dropped: %15"PRIu64,
			total_packets_tx,
			total_packets_rx,
			total_packets_dropped);

	printf("\nTX Speed = [%.2lf Mpps, %.2lf Gbps]. RX Speed = [%5.2lf Mpps, %.2lf Gbps]. latency count %10"PRIu64 ", average latency %lf",
			(double)total_packets_tx / diff,
			(double)(total_packets_tx * (PKTLEN + 20) * 8) / (1000 * diff),
			(double)total_packets_rx / diff,
			(double)(total_packets_rx * (PKTLEN + 20) * 8) / (1000 * diff),
			total_latency_cnt, (total_latency/total_latency_cnt)/(rte_get_tsc_hz()/1e6));
	printf("\n==========================================================\n");
	
	/* Get the hardware counters */
	struct rte_eth_stats stats;
	rte_eth_stats_get(0, &stats);
	printf("===== HW statistics: err packets-%lu,\treceived packets-%lu,\ttransmitted packets-%lu",
			stats.ierrors, stats.ipackets, stats.opackets);
	printf("\n==========================================================\n");

	/* Notify cores to reset their counters */
	for (core_id = 0; core_id < NUM_MAX_CORE; core_id ++) {
		loop_mark[core_id] = 1;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
static inline int
fast_rand(unsigned int *g_seed) {
    *g_seed = (214013 * *g_seed + 2531011);
    return (*g_seed >> 16) & 0x7FFF;
}

static void
tx_loop(context_t *context)
{
	unsigned int core_id = context->core_id;
	unsigned int queue_id = context->queue_id;
	unsigned int port_id = context->port_id;
	int nb_tx;
	char *pkt;
	struct ether_hdr *ethh;
	struct ipv4_hdr *iph;
	int i;
	unsigned int g_seed = core_id;

	unsigned long mask = 1 << core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), (cpu_set_t *)&mask) < 0) {
		printf("core id = %d\n", core_id);
		assert(0);
	}

	struct rte_mbuf *tx_packets[MBUF_NUM];
	setup_mbuf(port_id, mp, tx_packets);

#ifdef INCREASE_TTL
	unsigned long long freq_counter = 0;
#endif

	printf(">>>>>>>> TX thread running on core %d, queue %d\n", core_id, queue_id);
	core_statistics[core_id].enable = 1;

	for (;;) {
#ifdef INCREASE_TTL
		freq_counter++;
		if (freq_counter % (TIMER_MILLISECOND) == 0) {
			ttl++;
			freq_counter = 0;
			setup_mbuf(port_id, mp, tx_packets);
			printf("ttl changed to %d\n", ttl);
		}
#endif
#if 1
		for (i = 0; i < MBUF_NUM; i ++) {
			pkt = rte_pktmbuf_mtod(tx_packets[i], char *);
			ethh = (struct ether_hdr *)pkt;
			iph = (struct ipv4_hdr *)((unsigned char *)ethh + sizeof(struct ether_hdr));
			iph->dst_addr = fast_rand(&g_seed);
		}
#endif

		nb_tx = rte_eth_tx_burst(port_id, queue_id, tx_packets, MBUF_NUM);
		if (loop_mark[core_id] == 1) {
			core_statistics[core_id].tx = 0;
			core_statistics[core_id].dropped = 0;
			loop_mark[core_id] = 0;
		}

		core_statistics[core_id].tx += nb_tx;
		if (unlikely(nb_tx < MBUF_NUM)) {
			core_statistics[core_id].dropped += MBUF_NUM - nb_tx;
		}
	}
}

static void
rx_loop(int core_id, int queue_id)
{
	uint64_t prev_tsc, diff_tsc, cur_tsc, timer_tsc;

	unsigned long mask = 1 << core_id;
	if (sched_setaffinity(0, sizeof(unsigned long), (cpu_set_t *)&mask) < 0) {
		assert(0);
	}

	printf("RX thread running on core %d, queue %d\n", core_id, queue_id);
	core_statistics[core_id].enable = 1;

	while (1) {
		cur_tsc = rte_rdtsc();
		diff_tsc = cur_tsc - prev_tsc;

		/* if timer is enabled */
		if (timer_period > 0) {
			/* advance the timer */
			timer_tsc += diff_tsc;
			/* if timer has reached its timeout */
			if (unlikely(timer_tsc >= (uint64_t) timer_period)) {
				/* do this only on master core */
				if (queue_id == 0) {
					print_stats();
					/* reset the timer */
					timer_tsc = 0;
				}
			}
		}
		prev_tsc = cur_tsc;
	}
}

int main(int argc, char **argv)
{
	if (argc == 1) {
		PKTLEN = 64;
	} else if (argc == 2) {
		PKTLEN = atoi(argv[1]);
	} else {
		printf("Please specify pkt size as the argument\n");
		exit(0);
	}
	printf("======= Packet size %d =======\n", PKTLEN);

	uint8_t port_id = 0, nb_ports;
	struct rte_eth_conf port_conf;
	int ret;

	int t_argc = 5;
	const char *t_argv[] = {"./build/basicfwd", "-c", "0x3f", "-n", "4"};

	ret = rte_eal_init(t_argc, t_argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

	nb_ports = rte_eth_dev_count();
	assert(nb_ports == 1);

	init_mempool(&mp);

	port_conf_default.rxmode = rx_mode;
	port_conf_default.txmode = tx_mode;

	port_conf = port_conf_default;
	for (int i = 0; i < PORT_NUM; i++) {
		ret = rte_eth_dev_configure(i, NUM_RX_QUEUE, NUM_TX_QUEUE, &port_conf);
		if (ret != 0)
			return ret;
	}

	for (int i = 0; i < NUM_RX_QUEUE; i++) {
		ret = rte_eth_rx_queue_setup(port_id, i, RX_RING_SIZE,
				rte_eth_dev_socket_id(port_id), NULL, mp);
		if (ret < 0)
			return ret;
	}

	for (int i = 0; i < NUM_TX_QUEUE; i++) {
		ret = rte_eth_tx_queue_setup(port_id, i, TX_RING_SIZE,
				rte_eth_dev_socket_id(port_id), NULL);
		if (ret < 0)
			return ret;
	}

	ret = rte_eth_dev_start(port_id);
	if (ret < 0)
		return ret;

	rte_eth_promiscuous_enable(port_id);
	
	pthread_t tid;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

	context_t *context;

	int i;
	for (i = 0; i < NUM_TX_QUEUE; i ++) {
		context = (context_t *) malloc (sizeof(context_t));
		context->core_id = i;
		context->queue_id = i;
		context->port_id = 0;
		if (pthread_create(&tid, &attr, (void *)tx_loop, (void *)context) != 0) {
			perror("pthread_create error!!\n");
		}
	}

	rx_loop(i, 0);
	return 0;
}
