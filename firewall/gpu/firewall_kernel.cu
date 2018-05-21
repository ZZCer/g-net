#include <stdio.h>
#include <cuda_runtime.h>
#include "firewall_kernel.h"
#include <gpu_packet.h>

extern struct portTreeNode *dev_srcPortTree;
extern struct portTreeNode *dev_desPortTree;
extern struct trieAddrNode *dev_srcAddrTrie;
extern struct trieAddrNode *dev_desAddrTrie;
extern unsigned int *dev_protocolHash;

extern "C" __global__ void firewall_gpu(struct trieAddrNode *srcAddrTrie, struct trieAddrNode *desAddrTrie, unsigned int *protocolHash, 
				struct portTreeNode *srcPortTree, struct portTreeNode *desPortTree,
				unsigned int *res, gpu_packet_t **pkts, int pcktCount)
{
	int tid = threadIdx.x +  blockIdx.x * blockDim.x;

	for (; tid < pcktCount; tid += gridDim.x * blockDim.x) {
		int idx = 0;
		int resid = tid << 2;
		res[resid] = 0;
		res[resid+1] = 0;
		res[resid+2] = 0;
		res[resid+3] = 0;

		//***********
		//for srcAddr
		for (int i = 1; i <= 32; i++) {
			unsigned int tmp = pkts[tid]->src_addr;
			tmp = tmp >> (32-i);

			res[resid] = (res[resid] | srcAddrTrie[idx].matchRules[0]);
			res[resid+1] = (res[resid+1] | srcAddrTrie[idx].matchRules[1]);
			res[resid+2] = (res[resid+2] | srcAddrTrie[idx].matchRules[2]);
			res[resid+3] = (res[resid+3] | srcAddrTrie[idx].matchRules[3]);

			if ((tmp % 2) == 0) {
				if (srcAddrTrie[idx].leftChild != 0) {
					idx = srcAddrTrie[idx].leftChild;
				} else {
					break;
				}
			} else {
				if (srcAddrTrie[idx].rightChild != 0) {
					idx = srcAddrTrie[idx].rightChild;
				} else {
					break;
				}
			}
		}

		//***********
		//for desAddr
		idx = 0;
		int resDesAddr[4] = {0};
		for (int i = 1; i <= 32; i++) {
			unsigned int tmp = pkts[tid]->dst_addr;
			tmp = tmp >> (32-i);
			resDesAddr[0] = (resDesAddr[0] | desAddrTrie[idx].matchRules[0]);
			resDesAddr[1] = (resDesAddr[1] | desAddrTrie[idx].matchRules[1]);
			resDesAddr[2] = (resDesAddr[2] | desAddrTrie[idx].matchRules[2]);
			resDesAddr[3] = (resDesAddr[3] | desAddrTrie[idx].matchRules[3]);

			if ((tmp % 2) == 0) {
				if (desAddrTrie[idx].leftChild != 0) {
					idx = desAddrTrie[idx].leftChild;
				} else {
					break;
				}
			} else {
				if (desAddrTrie[idx].rightChild != 0) {
					idx = desAddrTrie[idx].rightChild;
				} else {
					break;
				}
			}
		}

		res[resid] = (res[resid] & resDesAddr[0]);
		res[resid+1] = (res[resid+1] & resDesAddr[1]);
		res[resid+2] = (res[resid+2] & resDesAddr[2]);
		res[resid+3] = (res[resid+3] & resDesAddr[3]);

		//************
		//for protocol
		res[resid] = (res[resid] & protocolHash[pkts[tid]->proto_id]);
		res[resid+1] = (res[resid+1] & protocolHash[pkts[tid]->proto_id+4]);
		res[resid+2] = (res[resid+2] & protocolHash[pkts[tid]->proto_id+8]);
		res[resid+3] = (res[resid+3] & protocolHash[pkts[tid]->proto_id+12]);

		//************
		//for src port
		int srcPortQueue[RULESIZE] = {0};

		int headSrc = -1;
		int tailSrc = 0;   //queue size = tailSrc-headSrc.

		srcPortQueue[tailSrc++] = srcPortTree[0].endPort;
		headSrc++;

		int resSrcPort[4] = {0};

		while((tailSrc-headSrc) > 0) {   //when size > 0, same as queue is not empty,
			//same as there are node to be deal with.
			//headSrc is the node we are dealing with.
			if (pkts[tid]->src_port > srcPortTree[srcPortQueue[headSrc]].max) {
				headSrc++;
			} else if (pkts[tid]->src_port < srcPortTree[srcPortQueue[headSrc]].startPort) {
				if (srcPortTree[srcPortQueue[headSrc]].leftChild != 0) {
					srcPortQueue[tailSrc++] = srcPortTree[srcPortQueue[headSrc]].leftChild;
				}

				headSrc++;
			} else if (pkts[tid]->src_port <= srcPortTree[srcPortQueue[headSrc]].endPort) {
				resSrcPort[0] = resSrcPort[0] | srcPortTree[srcPortQueue[headSrc]].matchRules[0];
				resSrcPort[1] = resSrcPort[1] | srcPortTree[srcPortQueue[headSrc]].matchRules[1];
				resSrcPort[2] = resSrcPort[2] | srcPortTree[srcPortQueue[headSrc]].matchRules[2];
				resSrcPort[3] = resSrcPort[3] | srcPortTree[srcPortQueue[headSrc]].matchRules[3];

				if (srcPortTree[srcPortQueue[headSrc]].leftChild != 0) {
					srcPortQueue[tailSrc++] = srcPortTree[srcPortQueue[headSrc]].leftChild;
				}
				if (srcPortTree[srcPortQueue[headSrc]].rightChild != 0) {
					srcPortQueue[tailSrc++] = srcPortTree[srcPortQueue[headSrc]].rightChild;
				}

				headSrc++;
			} else {
				if (srcPortTree[srcPortQueue[headSrc]].leftChild != 0) {
					srcPortQueue[tailSrc++] = srcPortTree[srcPortQueue[headSrc]].leftChild;
				}

				if (srcPortTree[srcPortQueue[headSrc]].rightChild != 0) {    
					srcPortQueue[tailSrc++] = srcPortTree[srcPortQueue[headSrc]].rightChild;
				}

				headSrc++;
			}
		}

		res[resid] = (res[resid] & resSrcPort[0]);
		res[resid+1] = (res[resid+1] & resSrcPort[1]);
		res[resid+2] = (res[resid+2] & resSrcPort[2]);
		res[resid+3] = (res[resid+3] & resSrcPort[3]);

		//************
		//for des port
		int desPortQueue[RULESIZE] = {0};

		int headDes = -1;
		int tailDes = 0;   //queue size = tailDes-headDes.

		desPortQueue[tailDes++] = desPortTree[0].endPort;

		headDes++;

		int resDesPort[4] = {0};

		while((tailDes-headDes) > 0) {   //when size > 0, same as queue is not empty,
			//same as there are node to be deal with.
			//headDes is the node we are dealing with.
			if (pkts[tid]->dst_port > desPortTree[desPortQueue[headDes]].max) {
				headDes++;
			} else if (pkts[tid]->dst_port < desPortTree[desPortQueue[headDes]].startPort) {
				if (desPortTree[desPortQueue[headDes]].leftChild != 0) {
					desPortQueue[tailDes++] = desPortTree[desPortQueue[headDes]].leftChild;
				}

				headDes++;
			} else if (pkts[tid]->dst_port <= desPortTree[desPortQueue[headDes]].endPort) {
				resDesPort[0] = resDesPort[0] | desPortTree[desPortQueue[headDes]].matchRules[0];
				resDesPort[1] = resDesPort[1] | desPortTree[desPortQueue[headDes]].matchRules[1];
				resDesPort[2] = resDesPort[2] | desPortTree[desPortQueue[headDes]].matchRules[2];
				resDesPort[3] = resDesPort[3] | desPortTree[desPortQueue[headDes]].matchRules[3];

				if (desPortTree[desPortQueue[headDes]].leftChild != 0) {
					desPortQueue[tailDes++] = desPortTree[desPortQueue[headDes]].leftChild;
				} if (desPortTree[desPortQueue[headDes]].rightChild != 0) {
					desPortQueue[tailDes++] = desPortTree[desPortQueue[headDes]].rightChild;
				}

				headDes++;
			} else {
				if (desPortTree[desPortQueue[headDes]].leftChild != 0) {
					desPortQueue[tailDes++] = desPortTree[desPortQueue[headDes]].leftChild;
				}

				if (desPortTree[desPortQueue[headDes]].rightChild != 0) {    
					desPortQueue[tailDes++] = desPortTree[desPortQueue[headDes]].rightChild;
				}

				headDes++;
			}
		}

		res[resid] = (res[resid] & resDesPort[0]);
		res[resid+1] = (res[resid+1] & resDesPort[1]);
		res[resid+2] = (res[resid+2] & resDesPort[2]);
		res[resid+3] = (res[resid+3] & resDesPort[3]);
	}
}
