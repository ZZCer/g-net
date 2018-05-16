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
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_tcp.h>
#include <rte_ether.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_framework.h"
#include "construct_rules.h"

#include "gpu/firewall_kernel.h"

#define NF_TAG "firewall"

extern struct portTreeNode *srcPortTree;
extern struct portTreeNode *desPortTree;
extern struct trieAddrNode *srcAddrTrie;
extern struct trieAddrNode *desAddrTrie;
extern unsigned int *protocolHash;

static CUdeviceptr dev_srcPortTree;
static CUdeviceptr dev_desPortTree;
static CUdeviceptr dev_srcAddrTrie;
static CUdeviceptr dev_desAddrTrie;
static CUdeviceptr dev_protocolHash;

typedef struct my_buf_s {
	/* Stores real data */
	uint64_t job_num;
	struct pcktFive *host_pkt_fives;
	unsigned int *host_res;
	CUdeviceptr dev_pkt_fives;
	CUdeviceptr dev_res;
} buf_t;

static void *init_host_buf(void)
{
	buf_t *buf = malloc(sizeof(buf_t));

	gcudaHostAlloc((void **)&(buf->host_pkt_fives), MAX_BATCH_SIZE * sizeof(struct pcktFive));
	gcudaHostAlloc((void **)&(buf->host_res), MAX_BATCH_SIZE * 4 * sizeof(unsigned int));

	gcudaMalloc(&(buf->dev_pkt_fives), MAX_BATCH_SIZE * sizeof(struct pcktFive));
	gcudaMalloc(&(buf->dev_res), MAX_BATCH_SIZE * 4 * sizeof(unsigned int));

	return buf;
}

static inline void user_batch_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	if (!onvm_pkt_is_ipv4(pkt))
		rte_exit(EXIT_FAILURE, "Packet is not ipv4\n");

	struct ipv4_hdr *ip = onvm_pkt_ipv4_hdr(pkt);
	buf->host_pkt_fives[pkt_idx].srcAddr = ip->src_addr;
	buf->host_pkt_fives[pkt_idx].desAddr = ip->dst_addr;

	if (onvm_pkt_is_tcp(pkt)) {
		struct tcp_hdr *tcp = onvm_pkt_tcp_hdr(pkt);
		buf->host_pkt_fives[pkt_idx].protocol = IP_PROTOCOL_TCP;
		buf->host_pkt_fives[pkt_idx].srcPort = tcp->src_port;
		buf->host_pkt_fives[pkt_idx].desPort = tcp->dst_port;
	} else if(onvm_pkt_is_udp(pkt)) {
		struct udp_hdr *udp = onvm_pkt_udp_hdr(pkt);
		buf->host_pkt_fives[pkt_idx].protocol = IP_PROTOCOL_UDP;
		buf->host_pkt_fives[pkt_idx].srcPort = udp->src_port;
		buf->host_pkt_fives[pkt_idx].desPort = udp->dst_port;
	} else {
		rte_exit(EXIT_FAILURE, "Packet is neither TCP or UDP\n");
	}

}

static inline void user_post_func(void *cur_buf, struct rte_mbuf *pkt, int pkt_idx)
{
	buf_t *buf = (buf_t *)cur_buf;

	/* Drops the packet if malicious content is detected */
	if (buf->host_res[pkt_idx] != 0) {
		struct onvm_pkt_meta *meta;
		meta = onvm_get_pkt_meta((struct rte_mbuf *)pkt);
		meta->action = ONVM_NF_ACTION_DROP;
		RTE_LOG(INFO, APP, "Packet is dropped by the Firewall\n");
	}
}

static void user_gpu_htod(void *cur_buf, int job_num)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyHtoD(buf->dev_pkt_fives, buf->host_pkt_fives, job_num * sizeof(struct pcktFive));
}

static void user_gpu_dtoh(void *cur_buf, int job_num)
{
	buf_t *buf = (buf_t *)cur_buf;
	gcudaMemcpyDtoH(buf->host_res, buf->dev_res, job_num * 4 * sizeof(unsigned int));
}

static void user_gpu_set_arg(void *cur_buf, void *arg_buf, void *arg_info, int job_num)
{
	uint64_t *info = (uint64_t *)arg_info;
	buf_t *buf = (buf_t *)cur_buf;

	uint64_t arg_num = 8;
	uint64_t offset = 0;

	info[0] = arg_num;

	info[1] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(dev_srcAddrTrie), sizeof(dev_srcAddrTrie));
	offset += sizeof(dev_srcAddrTrie);

	info[2] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(dev_desAddrTrie), sizeof(dev_desAddrTrie));
	offset += sizeof(dev_desAddrTrie);

	info[3] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(dev_protocolHash), sizeof(dev_protocolHash));
	offset += sizeof(dev_protocolHash);
	
	info[4] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(dev_srcPortTree), sizeof(dev_srcPortTree));
	offset += sizeof(dev_srcPortTree);

	info[5] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(dev_desPortTree), sizeof(dev_desPortTree));
	offset += sizeof(dev_desPortTree);

	info[6] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_res), sizeof(buf->dev_res));
	offset += sizeof(buf->dev_res);
	
	info[7] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(buf->dev_pkt_fives), sizeof(buf->dev_pkt_fives));
	offset += sizeof(buf->dev_pkt_fives);
	
	info[8] = offset;
	rte_memcpy((uint8_t *)arg_buf + offset, &(job_num), sizeof(job_num));
	offset += sizeof(job_num);
}

static void init_main(void)
{
	/* Read rules and construct rules */
	struct fwRule *rules = (struct fwRule *)malloc(RULESIZE * sizeof(struct fwRule));
	construct_rules(rules);

	struct portTreeNode *host_srcPortTree;
	struct portTreeNode *host_desPortTree;
	struct trieAddrNode *host_srcAddrTrie;
	struct trieAddrNode *host_desAddrTrie;
	unsigned int *host_protocolHash;

	/* allocate the host memory */
	gcudaAllocSize(MAX_BATCH_SIZE * sizeof(struct pcktFive) // input buffer
			+ MAX_BATCH_SIZE * 4 * sizeof(unsigned int), 				// output result
			SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode)
			+ DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode)
			+ SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode)
			+ DES_PORT_TREE_SIZE * sizeof(struct portTreeNode)
			+ PROTOCOL_HASH_SIZE * 4 * sizeof(int));


	gcudaHostAlloc((void **)&(host_srcAddrTrie), SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	gcudaHostAlloc((void **)&(host_desAddrTrie), DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	gcudaHostAlloc((void **)&(host_srcPortTree), SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	gcudaHostAlloc((void **)&(host_desPortTree), DES_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	gcudaHostAlloc((void **)&(host_protocolHash), PROTOCOL_HASH_SIZE * 4 * sizeof(int));

	rte_memcpy(host_srcAddrTrie, srcAddrTrie, SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	rte_memcpy(host_desAddrTrie, desAddrTrie, DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	rte_memcpy(host_srcPortTree, srcPortTree, SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	rte_memcpy(host_desPortTree, desPortTree, DES_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	rte_memcpy(host_protocolHash, protocolHash, PROTOCOL_HASH_SIZE * 4 * sizeof(int));

	gcudaMalloc(&(dev_srcAddrTrie), SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	gcudaMalloc(&(dev_desAddrTrie), DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	gcudaMalloc(&(dev_srcPortTree), SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	gcudaMalloc(&(dev_desPortTree), DES_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	gcudaMalloc(&(dev_protocolHash), PROTOCOL_HASH_SIZE * 4 * sizeof(int));

	/* Transfer to GPU with the data in the shared memory */
	gcudaMemcpyHtoD(dev_srcAddrTrie, host_srcAddrTrie, SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	gcudaMemcpyHtoD(dev_desAddrTrie, host_desAddrTrie, DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	gcudaMemcpyHtoD(dev_srcPortTree, host_srcPortTree, SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	gcudaMemcpyHtoD(dev_desPortTree, host_desPortTree, DES_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	gcudaMemcpyHtoD(dev_protocolHash, host_protocolHash, PROTOCOL_HASH_SIZE * 4 * sizeof(int));
}

static void init_gpu_schedule(void)
{
	/* Initialize the GPU info, onvm_framework_init should be performed before onvm_nflib_init */
	const char *module_file = "../firewall/gpu/firewall.ptx";
	const char *kernel_name = "firewall_gpu";
	onvm_framework_init(module_file, kernel_name, &(init_host_buf));

	double K1 = 0.136325;
	double B1 = 15.241;
	double K2 = 0.005968;
	double B2 = 9.3574;
	onvm_framework_install_kernel_perf_parameter(K1, B1, K2, B2);
}

int main(int argc, char *argv[])
{
	int arg_offset;

	/* Initialize nflib */
	if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG, NF_FIREWALL, &(init_gpu_schedule))) < 0)
		return -1;
	argc -= arg_offset;
	argv += arg_offset;

	/* ================================= */
	/* Initialize the app specific stuff */
	init_main();

	/* Initialization is done, start threads */
	onvm_framework_start_cpu(&(user_batch_func), &(user_post_func));

	onvm_framework_start_gpu(&(user_gpu_htod), &(user_gpu_dtoh), &(user_gpu_set_arg));

	printf("If we reach here, program is ending\n");
	return 0;
}
