#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <rte_memcpy.h>
#include <rte_ring.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>

#define RX_THREADS 8
#define TX_THREADS 8
#define RX_BATCH 32
#define TX_BATCH 64

rte_atomic64_t rx_count;
rte_atomic64_t tx_count;
struct rte_ring *queue;
struct rte_mempool *pktmbuf_pool;


static int init_mbuf_pools(void);
static int init_port(uint8_t port_num);

int rx_main(void *arg) {
    int queue_id = (int)(uintptr_t)arg;
    struct rte_mbuf *pkts[RX_BATCH];
    unsigned count = 0, sent;
    while (1) {
        count = rte_eth_rx_burst(0, queue_id, pkts, RX_BATCH);
        rte_atomic64_add(&rx_count, count);
        //sent = rte_eth_tx_burst(0, queue_id, pkts, count);
        //rte_atomic64_add(&tx_count, sent);
        sent = rte_ring_enqueue_burst(queue, (void **)pkts, count, NULL);
        for (; sent < count; sent++) rte_pktmbuf_free(pkts[sent]);
    }
}

int tx_main(void *arg) {
    int queue_id = (int)(uintptr_t)arg;
    struct rte_mbuf *pkts[TX_BATCH];
    unsigned dequeue, i;
    while (1) {
        dequeue = rte_ring_dequeue_burst(queue, (void **)pkts, TX_BATCH, NULL);
        //for (i = 0; i < dequeue; i++) rte_pktmbuf_free(pkts[i]);
        for (i = 0; i < dequeue; i += rte_eth_tx_burst(0, queue_id, pkts + i, dequeue - i));
        rte_atomic64_add(&tx_count, dequeue);
    }
}

int main(int argc, char **argv) {
    rte_eal_init(argc, argv);
    if (init_mbuf_pools() != 0)
        rte_exit(1, "failed to init mbufs\n");
    if (init_port(0) != 0)
        rte_exit(1, "failed to init port\n");
    queue = rte_ring_create("Q", 16384, rte_socket_id(), 0);
    if (!queue)
        rte_exit(1, "failed to create queue\n");
    rte_atomic64_init(&rx_count);
    rte_atomic64_init(&tx_count);

    unsigned lcore = rte_lcore_id();
    printf("main thread: %u\n", lcore);
    for (int i = 0; i < RX_THREADS; i++) {
        lcore = rte_get_next_lcore(lcore, 1, 1);
        printf("rx thread: %u\n", lcore);
        if (rte_eal_remote_launch(rx_main, (void *)(uintptr_t)i, lcore) == -EBUSY)
            rte_exit(1, "core busy\n");
    }
    for (int i = 0; i < TX_THREADS; i++) {
        lcore = rte_get_next_lcore(lcore, 1, 1);
        printf("tx thread: %u\n", lcore);
        if (rte_eal_remote_launch(tx_main, (void *)(uintptr_t)i, lcore) == -EBUSY)
            rte_exit(1, "core busy\n");
    }
    fflush(stdout);
    struct timespec last, now;
    while (1) {
        rte_atomic64_set(&rx_count, 0);
        rte_atomic64_set(&tx_count, 0);
        clock_gettime(CLOCK_MONOTONIC, &last);
        sleep(3);
        int64_t r = rte_atomic64_read(&rx_count);
        int64_t t = rte_atomic64_read(&tx_count);
        clock_gettime(CLOCK_MONOTONIC, &now);
        double us = (now.tv_sec - last.tv_sec) * 1000000.0 + (now.tv_nsec - last.tv_nsec) / 1000.0;
        printf("Rx %f Mpps, Tx %f Mpps\n", r / us, t / us);
        fflush(stdout);
    }
    return 0;
}


/* ========================================= */

static int
init_mbuf_pools(void) {
    const unsigned num_mbufs = 524288;

    pktmbuf_pool = rte_mempool_create("MBUF", num_mbufs,
            sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM + 2048,
            512,
            sizeof(struct rte_pktmbuf_pool_private), rte_pktmbuf_pool_init,
            NULL, rte_pktmbuf_init, NULL, rte_socket_id(), 0);

    return (pktmbuf_pool == NULL); /* 0  on success */
}

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

    const uint16_t rx_rings = RX_THREADS, tx_rings = TX_THREADS;
    const uint16_t rx_ring_size = 512;
    const uint16_t tx_ring_size = 512;

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

    return 0;
}
