#include <stdio.h>
#include <cuda_runtime.h>
#include "firewall_kernel.h"

extern struct portTreeNode *dev_srcPortTree;
extern struct portTreeNode *dev_desPortTree;
extern struct trieAddrNode *dev_srcAddrTrie;
extern struct trieAddrNode *dev_desAddrTrie;
extern unsigned int *dev_protocolHash;

extern "C" __global__ void firewall_gpu(struct trieAddrNode *srcAddrTrie, struct trieAddrNode *desAddrTrie, unsigned int *protocolHash, 
				struct portTreeNode *srcPortTree, struct portTreeNode *desPortTree,
				unsigned int *res, struct pcktFive *pcktFwFives, int pcktCount)
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
			unsigned int tmp = pcktFwFives[tid].srcAddr;
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
			unsigned int tmp = pcktFwFives[tid].desAddr;
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
		res[resid] = (res[resid] & protocolHash[pcktFwFives[tid].protocol]);
		res[resid+1] = (res[resid+1] & protocolHash[pcktFwFives[tid].protocol+4]);
		res[resid+2] = (res[resid+2] & protocolHash[pcktFwFives[tid].protocol+8]);
		res[resid+3] = (res[resid+3] & protocolHash[pcktFwFives[tid].protocol+12]);

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
			if (pcktFwFives[tid].srcPort > srcPortTree[srcPortQueue[headSrc]].max) {
				headSrc++;
			} else if (pcktFwFives[tid].srcPort < srcPortTree[srcPortQueue[headSrc]].startPort) {
				if (srcPortTree[srcPortQueue[headSrc]].leftChild != 0) {
					srcPortQueue[tailSrc++] = srcPortTree[srcPortQueue[headSrc]].leftChild;
				}

				headSrc++;
			} else if (pcktFwFives[tid].srcPort <= srcPortTree[srcPortQueue[headSrc]].endPort) {
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
			if (pcktFwFives[tid].desPort > desPortTree[desPortQueue[headDes]].max) {
				headDes++;
			} else if (pcktFwFives[tid].desPort < desPortTree[desPortQueue[headDes]].startPort) {
				if (desPortTree[desPortQueue[headDes]].leftChild != 0) {
					desPortQueue[tailDes++] = desPortTree[desPortQueue[headDes]].leftChild;
				}

				headDes++;
			} else if (pcktFwFives[tid].desPort <= desPortTree[desPortQueue[headDes]].endPort) {
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

extern "C" void firewall_kernel(struct pcktFive *dev_pcktFwFives, unsigned int *dev_res, int pcktCount, int block_num, int threads_per_blk)
{
	firewall_gpu<<<block_num, threads_per_blk>>>(dev_srcAddrTrie, dev_desAddrTrie, dev_protocolHash, dev_srcPortTree, dev_desPortTree, dev_res, dev_pcktFwFives, pcktCount);

/*
	for (int i = 0; i < N; i++) {
		//deal with res to know which rule(s) is matched
		int flag = 0;
		for (int j = 0; (j < 32) && (flag != 1) ; j++) {
			unsigned int tmp = res[i] >> (31-j);

			if ((tmp % 2) == 1) {
				printf("packet %d matches rule %d  \n", i, j);
				flag = 1;
				break;
			} else {
				//		printf("packet %d %d\n", i, pcktFwFives[i].desPort);
			}
		}
		for (int j = 32; (j < 64) && (flag != 1); j++) {
			unsigned int tmp = res[i+N] >> (63-j);
			if ((tmp % 2) == 1) {   

				printf("packet %d matches rule %d\n", i, j);
				flag = 1;
				break;

			} else {
				//printf("packet %d pack port end %d not\n", i, pcktFwFives[i].desPort);
			}

		}
		for (int j = 64; (j < 96) && (flag != 1); j++) {
			unsigned int tmp = res[i+N+N] >> (95-j);
			if ((tmp % 2) == 1) {
				printf("packet %d matches rule %d \n", i, j);
				flag = 1;
				break;
			} else {
				//printf("packet %d %d\n", i, pcktFwFives[i].desPort);
			}
		}
		for (int j = 96; (j < 128) && (flag != 1) ; j++) {
			unsigned int tmp = res[i+N+N+N] >> (127-j);
			if ((tmp % 2) == 1) {
				printf("packet %d matches rule %d \n", i, j);
				flag = 1;
				break;
			} else {
				//		printf("packet %d %d\n", i, pcktFwFives[i].srcPort);
			}
		}
	}
*/
	return ;
}
