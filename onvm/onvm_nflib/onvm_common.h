/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2016 George Washington University
 *            2015-2016 University of California Riverside
 *            2010-2014 Intel Corporation
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
 * onvm_common.h - shared data between host and NFs
 ********************************************************************/

#ifndef _COMMON_H_
#define _COMMON_H_

#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>
#include <rte_spinlock.h>
#include <stdint.h>
#include <assert.h>
#include <cuda.h>


/**********************************Macros*************************************/
// GPU Data Sharing
#define GPU_DATA_SHARE 0
#define GPU_PACKET_POOL_SIZE (4 * 1024 * 1024) // ~8GB data
#define GPU_PACKET_POOL_QUEUE_NAME "GPU_PACKET_POOL"

// New Switching
#define NEW_SWITCHING 1
#define BATCH_POOL_NAME "BATCH_POOL"
#define BATCH_POOL_SIZE 1024
#define BATCH_CACHE_SIZE 32
//#define USE_BATCH_SWITCHING 1
#define BATCH_QUEUE_FACTOR 4
#define STARVE_THRESHOLD 1000000

#define MZ_CLIENTS "MProc_clients"

#define MAX_PKT_LEN 1514

typedef struct gpu_packet_s {
       struct ipv4_hdr ipv4_hdr_data;
       union {
               struct tcp_hdr tcp_hdr_data;
               struct udp_hdr udp_hdr_data;
       };
       uint16_t payload_size;
       uint8_t payload[MAX_PKT_LEN];
} __attribute__((aligned(16))) gpu_packet_t;

#define PORT_TX_QUEUE "port_tx_q_%u"

/*****************************Original Below**********************************/

#define ONVM_NUM_RX_THREADS	4 /* Should be the same with the number of worker threads in a NF --- what??? rx threads are matched to the num of NICs*/
#define BQUEUE_SWITCH	1      /* Use BQueue to transfer packets */
//#define MEASURE_LATENCY	1     /* Measure the latency of each NF */

#define BATCH_DRIVEN_BUFFER_PASS 1 /* Launch kernel with exact the batch assigned by the Manager */
#define B_PARA	0.9 /* Pass buffer between threads before reaching the set batch size, 
						as there are costs in communication. (when BATCH_DRIVEN_BUFFER_PASS is diabled) */
//#define SYNC_MODE 1 /* Use sync for kernel and memcpy operations, NOTE: try this for the latency issue */
//#define STREAM_LAUNCH	1 /* Launch kernel per stream */
/* GRAPH_TIME print memcpy time and kernel time in the Manager, and the overall time in each NF,
 * Only print trace from Manager, print the trace with plot_solo.py in ../../script,
 * Print traces from both Manager and each NF, print with plot.py in ../../script. */
//#define GRAPH_TIME 1 /* Print all kernel and data transfer time in SYNC_MODE */
#if defined(GRAPH_TIME)
	#define SYNC_MODE 1
#endif

/* WARNING: the two parameters cannot be set too large. 
 * When set as 4 and 4096, the kernel launch time interval becomes extremely huge.
 * This would make the GPU sharing ineffective. FIXME: Don't know why. */
#define PRESET_BLK_NUM		1
#define PRESET_BATCH_SIZE	512

/* Enable/Disable functionality */
#define NETWORK 1
//#define NO_GPU 1
//#define NO_SCHEDULER 1

/* Different mode for experiments */
//#define UNCO_SHARE_GPU	1 /* All NFs allocate maximum resources of a GPU */
//#define FAIR_SHARE_GPU	1 /* Each NFs allocate the same share of a GPU */
//#define NOT_SHARE_GPU	1 /* Sequentially execute GPU kernels, not share */

//#define PERF_TEST_GPU 1 /* Measure the GPU execution time with a fixed batch size */
#if defined(PERF_TEST_GPU)
	#define BATCH_DRIVEN_BUFFER_PASS 1
	#define SCHED_BATCH_SIZE 512
#endif

//#define MEASURE_KERNEL_TIME 1 /* Measure kernel execution time. It needs sync after kernel, thus will degrade performance */
//#define MATCH_RX_THREAD 1    /* Same RX thread with the worker thread number of the first NF */
//#define RX_SPEED_TEST 1      /* Measuring RX speed at ONVM RX thread, do not need to launch NFs */
//#define RX_SPEED_TEST_2 1    /* Measuring RX speed at ONVM RX thread, before enqueue */
//#define NF_RX_SPEED_TEST 1   /* Drop packets as soon as NF receives them, measure queue performance */

/* Set the specific GPU hardware parameters */
//#define P100 1
#define TITANX 1
#if defined(P100)
	#define SM_TOTAL_NUM	56
#elif defined(TITANX)
	#define SM_TOTAL_NUM	28
#endif

/* GPU thread block size */
#define MAX_THREAD_PER_BLK 1024

/* Not read the pkt length from the packet to avoid stats overhead */
//#define FAST_STAT	1
#if defined(FAST_STAT)
	#define PKT_LEN 64
#endif

#define MAX_BATCH_SIZE 8192

#define NUM_BATCH_BUF 3

/*****************************************************************************/

/* Useful macros */
#define UNUSED(x) (void)(x)
#define GENERIC_MAX(x,y) ((x)>(y)?(x):(y))
#define GENERIC_MIN(x,y) ((x)<(y)?(x):(y))


// Number of packets to attempt to read from queue
#define PKT_READ_SIZE  ((uint16_t)32)
#define TX_RING_MIN_AVAIL_SLOTS	32	

#define GLOBAL_RSP_QUEUE 1024

#define ONVM_MAX_CHAIN_LENGTH 16  // the maximum chain length
#define MAX_CLIENTS 16            // total number of NFs allowed
#define MAX_SERVICES 16           // total number of unique services allowed
#define MAX_CLIENTS_PER_SERVICE 1 // max number of NFs per service.

#define MAX_CONCURRENCY_NUM	8     // max number of threads + streams per NF
#define MAX_ARG_SIZE		128
#define MAX_ARG_NUM			16
#define MAX_MODULE_FILE_LEN	100
#define MAX_KERNEL_NAME_LEN	50

#define ONVM_NF_ACTION_DROP	0   // drop packet
#define ONVM_NF_ACTION_NEXT	1   // to whatever the next action is configured by the SDN controller in the flow table
#define ONVM_NF_ACTION_TONF	2   // send to the NF specified in the argument field (assume it is on the same host)
#define ONVM_NF_ACTION_OUT	3    // send the packet out the NIC port set in the argument field
#define ONVM_NF_ACTION_LOOP	4  // For test, loop to the start of the service chain.

//extern uint8_t rss_symmetric_key[40];

/*
 * Define a client structure with all needed info, including
 * stats from the clients.
 */
struct client {
	struct rte_ring *response_q[MAX_CONCURRENCY_NUM];
	struct rte_ring *global_response_q;
	struct onvm_nf_info *info;
	uint16_t instance_id;

	struct rte_ring *rx_q_new;
	struct rte_ring *tx_q_new;

	double throughput_mpps; /* Throughput in mpps */
	double latency_us; /* latency in microseconds (us) */
	unsigned int avg_pkt_len;
	int nf_type; /* NF_BPS or NF_PPS */
	CUstream stream[MAX_CONCURRENCY_NUM];
	int recording[MAX_CONCURRENCY_NUM];
	CUevent gpu_start[MAX_CONCURRENCY_NUM];
	CUevent gpu_end[MAX_CONCURRENCY_NUM];
	CUevent kernel_start[MAX_CONCURRENCY_NUM];
	CUevent kernel_end[MAX_CONCURRENCY_NUM];

	double cost_time; /* Record the cost of GPU execution */

	/* Be the value of each CUDA stream */
	uint16_t threads_per_blk;
	uint16_t blk_num;
	uint32_t batch_size;

	uint16_t init;
	CUmodule module;
	CUfunction function;
	struct gpu_schedule_info *gpu_info;

	/* these stats hold how many packets the client will actually receive,
	 * and how many packets were dropped because the client's queue was full.
	 * The port-info stats, in contrast, record how many packets were received
	 * or transmitted on an actual NIC port.
	 */
	struct {
		rte_spinlock_t update_lock;
		// updated by switching/framework
		uint64_t rx;
		uint64_t rx_datalen;
		uint64_t tx;
		uint64_t tx_drop;
		uint64_t act_drop;

		uint64_t batch_size;
		double cpu_time;
		uint64_t batch_cnt;

		// recorded by stats
		struct timespec start;

		// updated by manager
		uint64_t htod_mem;
		uint64_t dtoh_mem;
		double gpu_time;
		double kernel_time;
		uint64_t kernel_cnt;
	} __attribute__ ((aligned (64))) stats;
};

struct onvm_pkt_meta {
	uint8_t action; /* Action to be performed */
	uint16_t destination; /* where to go next */
	uint16_t src; /* who processed the packet last */
	uint8_t chain_index; /*index of the current step in the service chain*/
};
static inline struct onvm_pkt_meta* onvm_get_pkt_meta(struct rte_mbuf* pkt) {
	return (struct onvm_pkt_meta*)&pkt->udata64;
}

static inline uint8_t onvm_get_pkt_chain_index(struct rte_mbuf* pkt) {
	return ((struct onvm_pkt_meta*)&pkt->udata64)->chain_index;
}

/*
 * Define a structure to describe one NF
 */
struct onvm_nf_info {
	uint16_t instance_id;
	uint16_t service_id;
	uint8_t status;
	const char *tag;
};

/*
 * Define a structure to describe a service chain entry
 */
struct onvm_service_chain_entry {
	uint16_t destination;
	uint8_t action;
};

struct onvm_service_chain {
	struct onvm_service_chain_entry sc[ONVM_MAX_CHAIN_LENGTH];
	uint8_t chain_length;
	int ref_cnt;
};



/* GPU scheduling info from the GPU */
/* ================================================== */
struct nf_req {
	volatile uint16_t type;
	volatile uint16_t instance_id; 
	volatile uint16_t thread_id;
	volatile uint16_t stream_id;
	volatile CUdeviceptr device_ptr;
	volatile uint32_t host_offset;
	volatile uint32_t size;
	volatile uint32_t value;
	struct nf_req *next;
};

struct nf_rsp {
	volatile int type;
	volatile int states;
	volatile int batch_size;
	volatile int instance_id;
	CUdeviceptr dev_ptr;
	int stream_id;
};

#define MAX_PARA_NUM 16

struct gpu_schedule_info {
	char module_file[MAX_MODULE_FILE_LEN];
	char kernel_name[MAX_KERNEL_NAME_LEN];

	/* schedule arguments, Latency = k * BATCH_SIZE + b */
	/* k1, b1: Parameters for linear equation between kernel execution time and batch size,
	 * L_{k} = k1 * B_{0} + b1, B_{0} is the batch size of a SM */
	/* k2, b2: Parameters for linear equation between data transfer time and batch size,
	 * L_{m} = k2 * B + b2, B is the total batch size, B = B_{0} * SM_NUM */
	double k1[MAX_PARA_NUM], k2[MAX_PARA_NUM], b1[MAX_PARA_NUM], b2[MAX_PARA_NUM];
	unsigned int pkt_size[MAX_PARA_NUM];
	unsigned int line_start_batch[MAX_PARA_NUM];
	unsigned int para_num;

	char args[MAX_CONCURRENCY_NUM][MAX_ARG_SIZE];
	void *arg_info[MAX_CONCURRENCY_NUM][MAX_ARG_NUM + 1];

	unsigned int stream_num;
	int init;

	uint16_t latency_us;

	int launch_tx_thread;
	int launch_worker_thread;
};



/* Request and response types */
#define REQ_HOST_MALLOC				0
#define REQ_GPU_MALLOC				1
#define REQ_GPU_MEMCPY_HTOD_SYNC	2
#define REQ_GPU_MEMCPY_HTOD_ASYNC	3
#define REQ_GPU_MEMCPY_DTOH_SYNC	4
#define REQ_GPU_MEMCPY_DTOH_ASYNC	5
#define REQ_GPU_LAUNCH_STREAM_ASYNC 6
#define REQ_GPU_LAUNCH_ALL_STREAM	7
#define REQ_GPU_SYNC				8
#define REQ_GPU_MEMFREE				9
#define REQ_GPU_MEMSET				10
#define REQ_GPU_SYNC_STREAM			11
#define REQ_GPU_RECORD_START		12

#define RSP_HOST_MALLOC				0
#define RSP_GPU_MALLOC				1
#define RSP_GPU_MEMCPY_HTOD_SYNC	2
#define RSP_GPU_MEMCPY_DTOH_SYNC	3
#define RSP_GPU_GLOBAL_SYNC			4
#define RSP_GPU_KERNEL_SYNC			5
#define RSP_GPU_SYNC_STREAM			6

#define RSP_SUCCESS		0
#define RSP_FAILURE		1

/* Network function types */
#define NF_ROUTER		1
#define NF_FIREWALL		2
#define NF_NIDS			3
#define NF_IPSEC		4
#define NF_SOMEONE		5
#define NF_PKTGEN		6
#define NF_RAW			7
#define NF_FRAMEWORK	8
#define NF_END			9

/* Scheduling strategies */
#define NO_SCHED	0
#define NF_PPS		1
#define NF_BPS		2

struct nf_str {
	char const *name;
	int id;
	int type;
};

static struct nf_str nf_id_to_name[] =
{
	{ "None", 0, NO_SCHED },
	{ "Router", 1, NF_PPS },
	{ "Firewall", 2, NF_PPS },
	{ "NIDS", 3, NF_BPS },
	{ "IPSec", 4, NF_BPS },
	{ "Someone", 5, NO_SCHED },
	{ "Pktgen", 6, NO_SCHED },
	{ "NF Raw", 7, NO_SCHED },
	{ "NF Framework", 8, NO_SCHED },
	{ NULL, -1, NO_SCHED }
};

static inline const char *get_nf_name(int service_id)
{
	int i = 0;
	while ((nf_id_to_name[i].id != service_id) & (nf_id_to_name[i].id != -1)) {
		i ++;
	}

	if (nf_id_to_name[i].id == service_id)
		return (const char *)nf_id_to_name[i].name;
	else
		return (const char *)"NF name not found!";
}

static inline int get_nf_type(int service_id)
{
	int i = 0;
	while ((nf_id_to_name[i].id != service_id) & (nf_id_to_name[i].id != -1)) {
		i ++;
	}

	if (nf_id_to_name[i].id == service_id)
		return nf_id_to_name[i].type;
	else
		return 0;
}

/* ================================================== */


/* define common names for structures shared between server and client */
#define MP_CLIENT_RXQ_NAME "MProc_Client_%u_%u_RX"
#define MP_CLIENT_TXQ_NAME "MProc_Client_%u_TX"
#define MP_CLIENT_BQ_RX_NAME "MProc_BQueue_%u_%u_RX"
#define MP_CLIENT_BQ_TX_NAME "MProc_BQueue_%u_%u_TX"
#define PKTMBUF_POOL_NAME "MProc_pktmbuf_pool"
#define MZ_PORT_INFO "MProc_port_info"
#define MZ_CLIENT_INFO "MProc_client_info"
#define MZ_SCP_INFO "MProc_scp_info"
#define MZ_FTP_INFO "MProc_ftp_info"
#define MZ_GPU_INFO_NAME "MProc_GPU_info_%u"


/* common names for NF states */
#define _NF_QUEUE_NAME "NF_INFO_QUEUE"
#define _NF_MEMPOOL_NAME "NF_INFO_MEMPOOL"

#define _NF_REQUEST_QUEUE_NAME "NF_REQUEST_QUEUE"
#define _NF_RESPONSE_QUEUE_NAME "MProc_RSP_%u_%u"
#define _NF_BUF_NAME "MProc_Buf_%u"


#define _NF_REQUEST_MEMPOOL_NAME "NF_REQ_MEMPOOL"
#define _NF_RESPONSE_MEMPOOL_NAME "NF_RSP_MEMPOOL"

#define NF_WAITING_FOR_ID 0     // First step in startup process, doesn't have ID confirmed by manager yet
#define NF_STARTING 1           // When a NF is in the startup process and already has an id
#define NF_RUNNING 2            // Running normally
#define NF_PAUSED  3            // NF is not receiving packets, but may in the future
#define NF_STOPPED 4            // NF has stopped and in the shutdown process
#define NF_ID_CONFLICT 5        // NF is trying to declare an ID already in use
#define NF_NO_IDS 6             // There are no available IDs for this NF

#define NF_NO_ID -1

/*
 * Given the rx queue name template above, get the queue name
 */
static inline const char *
get_rx_queue_name(unsigned client_id, unsigned thread_id) {
	/* buffer for return value. Size calculated by %u being replaced
	 * by maximum 3 digits (plus an extra byte for safety) */
	static char buffer[sizeof(MP_CLIENT_RXQ_NAME) + 2];

	snprintf(buffer, sizeof(buffer) - 1, MP_CLIENT_RXQ_NAME, client_id, thread_id);
	return buffer;
}

/*
 * Given the tx queue name template above, get the queue name
 */
static inline const char *
get_tx_queue_name(unsigned id) {
	/* buffer for return value. Size calculated by %u being replaced
	 * by maximum 3 digits (plus an extra byte for safety) */
	static char buffer[sizeof(MP_CLIENT_TXQ_NAME) + 2];

	snprintf(buffer, sizeof(buffer) - 1, MP_CLIENT_TXQ_NAME, id);
	return buffer;
}

/* bqueue name - rx */
static inline const char *
get_rx_bq_name(unsigned client_id, unsigned thread_id) {
	static char buffer[sizeof(MP_CLIENT_BQ_RX_NAME) + 2];
	snprintf(buffer, sizeof(buffer) - 1, MP_CLIENT_BQ_RX_NAME, client_id, thread_id);
	return buffer;
}

/* bqueue name - tx */
static inline const char *
get_tx_bq_name(unsigned client_id, unsigned thread_id) {
	static char buffer[sizeof(MP_CLIENT_BQ_TX_NAME) + 2];
	snprintf(buffer, sizeof(buffer) - 1, MP_CLIENT_BQ_TX_NAME, client_id, thread_id);
	return buffer;
}

static inline const char *
get_rsp_queue_name(unsigned instance_id, unsigned thread_id) {
	static char buffer[sizeof(_NF_RESPONSE_QUEUE_NAME) + 2];

	snprintf(buffer, sizeof(buffer) - 1, _NF_RESPONSE_QUEUE_NAME, instance_id, thread_id);
	return buffer;
}

static inline const char *
get_buf_name(unsigned instance_id) {
	static char buffer[sizeof(_NF_BUF_NAME) + 2];
	snprintf(buffer, sizeof(buffer) - 1, _NF_BUF_NAME, instance_id);
	return buffer;
}

static inline const char *
get_gpu_info_name(unsigned instance_id) {
	static char buffer[sizeof(MZ_GPU_INFO_NAME) + 2];
	snprintf(buffer, sizeof(buffer) - 1, MZ_GPU_INFO_NAME, instance_id);
	return buffer;
}

static inline const char *
get_port_tx_queue_name(uint16_t port_id) {
       static char buffer[sizeof(PORT_TX_QUEUE) + 2];
       snprintf(buffer, sizeof(buffer) - 1, PORT_TX_QUEUE, port_id);
       return buffer;
}

#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1

#endif  // _COMMON_H_
