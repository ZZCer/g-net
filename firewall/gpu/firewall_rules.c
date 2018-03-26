#include <cuda_runtime.h>
#include <stdlib.h>

#include "firewall_kernel.h"

struct portTreeNode *srcPortTree;
struct portTreeNode *desPortTree;
struct trieAddrNode *srcAddrTrie;
struct trieAddrNode *desAddrTrie;
unsigned int *protocolHash;

struct portTreeNode *dev_srcPortTree;
struct portTreeNode *dev_desPortTree;
struct trieAddrNode *dev_srcAddrTrie;
struct trieAddrNode *dev_desAddrTrie;
unsigned int *dev_protocolHash;


static void build_src_trie_node(struct fwRule rules[], struct trieAddrNode srcAddrTrie[], int *srcAddrTrieCount, int i)
{
	int idx = 0;
	//to find the right node
	int j;
	for (j = 1; j <= rules[i].rule.srcMask; j++) {
		unsigned int tmp = rules[i].rule.srcAddr;
		tmp = tmp >> (32 - j);

		if ((tmp % 2) == 0) {
			if (srcAddrTrie[idx].leftChild == 0) {
				srcAddrTrie[idx].leftChild = (*srcAddrTrieCount)++;
				idx = srcAddrTrie[idx].leftChild;
			} else {
				idx = srcAddrTrie[idx].leftChild;
			}
		} else {
			if (srcAddrTrie[idx].rightChild == 0) {
				srcAddrTrie[idx].rightChild = (*srcAddrTrieCount)++;
				idx = srcAddrTrie[idx].rightChild;
			} else {
				idx = srcAddrTrie[idx].rightChild;
			}
		}
	}

	//add rules in src addr trie
	if (i < 32) {
		unsigned int tmp = 1 << (31 - i);
		srcAddrTrie[idx].matchRules[0] = srcAddrTrie[idx].matchRules[0] | tmp;
	} else if (i < 64) {
		unsigned int tmp = 1 << (63 - i);
		srcAddrTrie[idx].matchRules[1] = srcAddrTrie[idx].matchRules[1] | tmp;
	} else if (i < 96) {
		unsigned int tmp = 1 << (95 - i);
		srcAddrTrie[idx].matchRules[2] = srcAddrTrie[idx].matchRules[2] | tmp;
	} else {
		unsigned int tmp = 1 << (127 - i);
		srcAddrTrie[idx].matchRules[3] = srcAddrTrie[idx].matchRules[3] | tmp;
	}
}

static void build_des_trie_node(struct fwRule rules[], struct trieAddrNode desAddrTrie[], int *desAddrTrieCount, int i)
{
	int idx = 0;
	//to find the right node
	int j;
	for (j = 1; j <= rules[i].rule.desMask; j++) {
		unsigned int tmp = rules[i].rule.desAddr;
		tmp = tmp >> (32 - j);

		if ((tmp % 2) == 0) {
			if (desAddrTrie[idx].leftChild == 0) {
				desAddrTrie[idx].leftChild = (*desAddrTrieCount)++;
				idx = desAddrTrie[idx].leftChild;
			} else {
				idx = desAddrTrie[idx].leftChild;
			}
		} else {
			if (desAddrTrie[idx].rightChild == 0) {
				desAddrTrie[idx].rightChild = (*desAddrTrieCount)++;
				idx = desAddrTrie[idx].rightChild;
			} else {
				idx = desAddrTrie[idx].rightChild;
			}
		}
	}

	//add rules in src addr trie
	if (i < 32) {
		unsigned int tmp = 1 << (31 - i);
		desAddrTrie[idx].matchRules[0] = desAddrTrie[idx].matchRules[0] | tmp;
	} else if (i < 64) {
		unsigned int tmp = 1 << (63 - i);
		desAddrTrie[idx].matchRules[1] = desAddrTrie[idx].matchRules[1] | tmp;
	} else if (i < 96) {
		unsigned int tmp = 1 << (95 - i);
		desAddrTrie[idx].matchRules[2] = desAddrTrie[idx].matchRules[2] | tmp;
	} else {
		unsigned int tmp = 1 << (127 - i);
		desAddrTrie[idx].matchRules[3] = desAddrTrie[idx].matchRules[3] | tmp;
	}
}

static void color_src_tree(struct portTreeNode srcPortTree[], int *rootSrc, int idx)
{
	int grandpa; 
	int uncle = 0;
	int parent;

	//if parent is black, do nothing.
	while (srcPortTree[srcPortTree[idx].parent].color == RED) {
		parent = srcPortTree[idx].parent;
		grandpa = srcPortTree[parent].parent;

		if (parent == srcPortTree[grandpa].leftChild) {
			uncle = srcPortTree[grandpa].rightChild;
		} else if (parent == srcPortTree[grandpa].rightChild) {
			uncle = srcPortTree[grandpa].leftChild;
		}

		if (srcPortTree[uncle].color == RED) {
			srcPortTree[parent].color = BLACK;
			srcPortTree[uncle].color = BLACK;
			srcPortTree[grandpa].color = RED;
			idx = grandpa;
		} else {
			if ((idx == srcPortTree[parent].rightChild) && (parent == srcPortTree[grandpa].leftChild)) {
				//left_rotate(parent);
				int x = parent;
				int y = srcPortTree[x].rightChild;
				srcPortTree[x].rightChild = srcPortTree[y].leftChild;
				if (srcPortTree[y].leftChild != 0) {
					srcPortTree[srcPortTree[y].leftChild].parent = x;
				}
				srcPortTree[y].parent = srcPortTree[x].parent;
				if (*rootSrc == x) {
					*rootSrc = y;
				} else if (x == srcPortTree[srcPortTree[x].parent].leftChild) {
					srcPortTree[srcPortTree[x].parent].leftChild = y;
				} else {
					srcPortTree[srcPortTree[x].parent].rightChild = y;
				}
				srcPortTree[y].leftChild = x;
				srcPortTree[x].parent = y;
				//end of left_rotate

				idx = srcPortTree[idx].leftChild;
			} else if ((idx == srcPortTree[parent].leftChild) && (parent == srcPortTree[grandpa].rightChild)) {
				//right_rotate(parent);
				int x = parent;
				int y = srcPortTree[x].leftChild;
				srcPortTree[x].leftChild = srcPortTree[y].rightChild;
				if (srcPortTree[y].rightChild != 0) {
					srcPortTree[srcPortTree[y].rightChild].parent = x;
				}
				srcPortTree[y].parent = srcPortTree[x].parent;
				if (*rootSrc == x) {
					*rootSrc = y;
				} else if (x == srcPortTree[srcPortTree[x].parent].rightChild) {
					srcPortTree[srcPortTree[x].parent].rightChild = y;
				} else {
					srcPortTree[srcPortTree[x].parent].leftChild = y;
				}
				srcPortTree[y].rightChild = x;
				srcPortTree[x].parent = y;
				//end of right_rotate

				idx = srcPortTree[idx].rightChild;
			} else if ((idx == srcPortTree[parent].leftChild) && (parent == srcPortTree[grandpa].leftChild)) {
				srcPortTree[parent].color = BLACK;
				srcPortTree[grandpa].color = RED;
				//right_rotate(grandpa);
				int x = grandpa;
				int y = srcPortTree[x].leftChild;
				srcPortTree[x].leftChild = srcPortTree[y].rightChild;
				if (srcPortTree[y].rightChild != 0) {
					srcPortTree[srcPortTree[y].rightChild].parent = x;
				}
				srcPortTree[y].parent = srcPortTree[x].parent;
				if (*rootSrc == x) {
					*rootSrc = y;
				} else if (x == srcPortTree[srcPortTree[x].parent].rightChild) {
					srcPortTree[srcPortTree[x].parent].rightChild = y;
				} else {
					srcPortTree[srcPortTree[x].parent].leftChild = y;
				}
				srcPortTree[y].rightChild = x;
				srcPortTree[x].parent = y;
				//end of right_rotate
			} else if ((idx == srcPortTree[parent].rightChild) && (parent == srcPortTree[grandpa].rightChild)) {
				srcPortTree[parent].color = BLACK;
				srcPortTree[grandpa].color = RED;
				//left_rotate(grandpa);
				int x = grandpa;
				int y = srcPortTree[x].rightChild;
				srcPortTree[x].rightChild = srcPortTree[y].leftChild;
				if (srcPortTree[y].leftChild != 0) {
					srcPortTree[srcPortTree[y].leftChild].parent = x;
				}
				srcPortTree[y].parent = srcPortTree[x].parent;
				if (*rootSrc == x) {
					*rootSrc = y;
				} else if (x == srcPortTree[srcPortTree[x].parent].leftChild) {
					srcPortTree[srcPortTree[x].parent].leftChild = y;
				} else {
					srcPortTree[srcPortTree[x].parent].rightChild = y;
				}
				srcPortTree[y].leftChild = x;
				srcPortTree[x].parent = y;
				//end of left_rotate
			}
		}

		srcPortTree[*rootSrc].color = BLACK;
	}//end of while
}

static void insert_src_tree_node(struct portTreeNode srcPortTree[], struct fwRule rules[], int *srcPortTreeCount, int i, int *rootSrc)
{
	int parent;

	int idx = *rootSrc;  //start from root, not 0 or 1
	int flag = 0;
	srcPortTree[0].color = BLACK;

	//to find the right place and put the rule in the tree
	while(flag == 0) {
		//first: is root nil?
		if (*rootSrc == 0) {
			//root is nil. put the rule in the root node.
			idx = (*srcPortTreeCount)++;

			srcPortTree[idx].color = BLACK;
			srcPortTree[idx].startPort = rules[i].rule.srcPortStart;
			srcPortTree[idx].endPort = rules[i].rule.srcPortEnd;

			if (i < 32) {
				unsigned int tmp = 1 << (31 - i);
				srcPortTree[idx].matchRules[0] = srcPortTree[idx].matchRules[0] | tmp;
			} else if (i < 64) {
				unsigned int tmp = 1 << (63 - i);
				srcPortTree[idx].matchRules[1] = srcPortTree[idx].matchRules[1] | tmp;
			} else if (i < 96) {
				unsigned int tmp = 1 << (95 - i);
				srcPortTree[idx].matchRules[2] = srcPortTree[idx].matchRules[2] | tmp;
			} else {
				unsigned int tmp = 1 << (127 - i);
				srcPortTree[idx].matchRules[3] = srcPortTree[idx].matchRules[3] | tmp;
			}

			srcPortTree[idx].max = rules[i].rule.srcPortEnd;

			*rootSrc = idx;
			flag = 2;
		} else {
			//root is not nil.
			if (rules[i].rule.srcPortStart < srcPortTree[idx].startPort) {
				//packet's < node's: go to left child

				parent = idx;
				idx = srcPortTree[idx].leftChild;

				if (idx == 0) {
					//we find what we need. new a idx node.
					idx = (*srcPortTreeCount)++;

					srcPortTree[idx].color = RED;
					srcPortTree[idx].startPort = rules[i].rule.srcPortStart;
					srcPortTree[idx].endPort = rules[i].rule.srcPortEnd;

					if (i < 32) {
						unsigned int tmp = 1 << (31 - i);
						srcPortTree[idx].matchRules[0] = srcPortTree[idx].matchRules[0] | tmp;
					} else if (i < 64) {
						unsigned int tmp = 1 << (63 - i);
						srcPortTree[idx].matchRules[1] = srcPortTree[idx].matchRules[1] | tmp;
					} else if (i < 96) {
						unsigned int tmp = 1 << (95 - i);
						srcPortTree[idx].matchRules[2] = srcPortTree[idx].matchRules[2] | tmp;
					} else {
						unsigned int tmp = 1 << (127 - i);
						srcPortTree[idx].matchRules[3] = srcPortTree[idx].matchRules[3] | tmp;
					}

					srcPortTree[idx].max = rules[i].rule.srcPortEnd;
					srcPortTree[idx].parent = parent;

					srcPortTree[parent].leftChild = idx;

					flag = 1;
				}
			} else if (rules[i].rule.srcPortStart > srcPortTree[idx].startPort) {
				parent = idx;
				idx = srcPortTree[idx].rightChild;

				if (idx == 0) {
					idx = (*srcPortTreeCount)++;

					srcPortTree[idx].color = RED;
					srcPortTree[idx].startPort = rules[i].rule.srcPortStart;
					srcPortTree[idx].endPort = rules[i].rule.srcPortEnd;

					if (i < 32) {
						unsigned int tmp = 1 << (31 - i);
						srcPortTree[idx].matchRules[0] = srcPortTree[idx].matchRules[0] | tmp;
					} else if (i < 64) {
						unsigned int tmp = 1 << (63 - i);
						srcPortTree[idx].matchRules[1] = srcPortTree[idx].matchRules[1] | tmp;
					} else if (i < 96) {
						unsigned int tmp = 1 << (95 - i);
						srcPortTree[idx].matchRules[2] = srcPortTree[idx].matchRules[2] | tmp;
					} else {
						unsigned int tmp = 1 << (127 - i);
						srcPortTree[idx].matchRules[3] = srcPortTree[idx].matchRules[3] | tmp;
					}

					srcPortTree[idx].max = rules[i].rule.srcPortEnd;
					srcPortTree[idx].parent = parent;

					srcPortTree[parent].rightChild = idx;

					flag = 1;
				}
			} else if (rules[i].rule.srcPortStart == srcPortTree[idx].startPort) {
				if (rules[i].rule.srcPortEnd == srcPortTree[idx].endPort) {

					if (i < 32) {
						unsigned int tmp = 1 << (31 - i);
						srcPortTree[idx].matchRules[0] = srcPortTree[idx].matchRules[0] | tmp;
					} else if (i < 64) {
						unsigned int tmp = 1 << (63 - i);
						srcPortTree[idx].matchRules[1] = srcPortTree[idx].matchRules[1] | tmp;
					} else if (i < 96) {
						unsigned int tmp = 1 << (95 - i);
						srcPortTree[idx].matchRules[2] = srcPortTree[idx].matchRules[2] | tmp;
					} else {
						unsigned int tmp = 1 << (127 - i);
						srcPortTree[idx].matchRules[3] = srcPortTree[idx].matchRules[3] | tmp;
					}
					flag = 2;
					//we dont need a new node. so it is no need to color.
					//no need to deal with max.
				} else if (rules[i].rule.srcPortEnd < srcPortTree[idx].endPort) {
					parent = idx;
					idx = srcPortTree[idx].leftChild;

					if (idx == 0) {
						//we find what we need. new a idx node.
						idx = (*srcPortTreeCount)++;

						srcPortTree[idx].color = RED;
						srcPortTree[idx].startPort = rules[i].rule.srcPortStart;
						srcPortTree[idx].endPort = rules[i].rule.srcPortEnd;

						if (i < 32) {
							unsigned int tmp = 1 << (31 - i);
							srcPortTree[idx].matchRules[0] = srcPortTree[idx].matchRules[0] | tmp;
						} else if (i < 64) {
							unsigned int tmp = 1 << (63 - i);
							srcPortTree[idx].matchRules[1] = srcPortTree[idx].matchRules[1] | tmp;
						} else if (i < 96) {
							unsigned int tmp = 1 << (95 - i);
							srcPortTree[idx].matchRules[2] = srcPortTree[idx].matchRules[2] | tmp;
						} else {
							unsigned int tmp = 1 << (127 - i);
							srcPortTree[idx].matchRules[3] = srcPortTree[idx].matchRules[3] | tmp;
						}

						srcPortTree[idx].max = rules[i].rule.srcPortEnd;
						srcPortTree[idx].parent = parent;

						srcPortTree[parent].leftChild = idx;

						flag = 1;
					}
				} else if (rules[i].rule.srcPortEnd > srcPortTree[idx].endPort) {
					parent = idx;
					idx = srcPortTree[idx].rightChild;

					if (idx == 0) {
						idx = (*srcPortTreeCount)++;

						srcPortTree[idx].color = RED;
						srcPortTree[idx].startPort = rules[i].rule.srcPortStart;
						srcPortTree[idx].endPort = rules[i].rule.srcPortEnd;

						if (i < 32) {
							unsigned int tmp = 1 << (31 - i);
							srcPortTree[idx].matchRules[0] = srcPortTree[idx].matchRules[0] | tmp;
						} else if (i < 64) {
							unsigned int tmp = 1 << (63 - i);
							srcPortTree[idx].matchRules[1] = srcPortTree[idx].matchRules[1] | tmp;
						} else if (i < 96) {
							unsigned int tmp = 1 << (95 - i);
							srcPortTree[idx].matchRules[2] = srcPortTree[idx].matchRules[2] | tmp;
						} else {
							unsigned int tmp = 1 << (127 - i);
							srcPortTree[idx].matchRules[3] = srcPortTree[idx].matchRules[3] | tmp;
						}

						srcPortTree[idx].max = rules[i].rule.srcPortEnd;
						srcPortTree[idx].parent = parent;

						srcPortTree[parent].rightChild = idx;

						flag = 1;
					}
				}
			}
		}
	}
	//end of insertion: without coloring. no bug(Apr.28)

	if (flag != 2) {
		color_src_tree(srcPortTree, rootSrc, idx);
	}
}

static void color_des_tree(struct portTreeNode desPortTree[], int *rootDes, int idx)
{
	//to color.

	//parent = desPortTree[idx].parent;
	//no need to calculate parent. when jumped out from the while, we already got the true parent.

	int grandpa; //= desPortTree[parent].parent;
	int uncle = 0;
	int parent;

	//if parent is black, do nothing.
	while (desPortTree[desPortTree[idx].parent].color == RED) {

		parent = desPortTree[idx].parent;
		grandpa = desPortTree[parent].parent;

		if (parent == desPortTree[grandpa].leftChild) {
			uncle = desPortTree[grandpa].rightChild;
		} else if (parent == desPortTree[grandpa].rightChild) {
			uncle = desPortTree[grandpa].leftChild;
		}


		if (desPortTree[uncle].color == RED) {
			desPortTree[parent].color = BLACK;
			desPortTree[uncle].color = BLACK;
			desPortTree[grandpa].color = RED;
			idx = grandpa;
		} else {
			if ((idx == desPortTree[parent].rightChild) && (parent == desPortTree[grandpa].leftChild)) {
				//left_rotate(parent);
				int x = parent;
				int y = desPortTree[x].rightChild;
				desPortTree[x].rightChild = desPortTree[y].leftChild;
				if (desPortTree[y].leftChild != 0)
					desPortTree[desPortTree[y].leftChild].parent = x;
				desPortTree[y].parent = desPortTree[x].parent;
				if (*rootDes == x)
					*rootDes = y;
				else if (x == desPortTree[desPortTree[x].parent].leftChild)
					desPortTree[desPortTree[x].parent].leftChild = y;
				else
					desPortTree[desPortTree[x].parent].rightChild = y;
				desPortTree[y].leftChild = x;
				desPortTree[x].parent = y;
				//end of left_rotate

				idx = desPortTree[idx].leftChild;
			} else if ((idx == desPortTree[parent].leftChild) && (parent == desPortTree[grandpa].rightChild)) {
				//right_rotate(parent);
				int x = parent;
				int y = desPortTree[x].leftChild;
				desPortTree[x].leftChild = desPortTree[y].rightChild;
				if (desPortTree[y].rightChild != 0)
					desPortTree[desPortTree[y].rightChild].parent = x;
				desPortTree[y].parent = desPortTree[x].parent;
				if (*rootDes == x)
					*rootDes = y;
				else if (x == desPortTree[desPortTree[x].parent].rightChild)
					desPortTree[desPortTree[x].parent].rightChild = y;
				else
					desPortTree[desPortTree[x].parent].leftChild = y;
				desPortTree[y].rightChild = x;
				desPortTree[x].parent = y;
				//end of right_rotate

				idx = desPortTree[idx].rightChild;
			} else if ((idx == desPortTree[parent].leftChild) && (parent == desPortTree[grandpa].leftChild)) {
				desPortTree[parent].color = BLACK;
				desPortTree[grandpa].color = RED;
				//right_rotate(grandpa);
				int x = grandpa;
				int y = desPortTree[x].leftChild;
				desPortTree[x].leftChild = desPortTree[y].rightChild;
				if (desPortTree[y].rightChild != 0)
					desPortTree[desPortTree[y].rightChild].parent = x;
				desPortTree[y].parent = desPortTree[x].parent;
				if (*rootDes == x)
					*rootDes = y;
				else if (x == desPortTree[desPortTree[x].parent].rightChild)
					desPortTree[desPortTree[x].parent].rightChild = y;
				else
					desPortTree[desPortTree[x].parent].leftChild = y;
				desPortTree[y].rightChild = x;
				desPortTree[x].parent = y;
				//end of right_rotate
			} else if ((idx == desPortTree[parent].rightChild) && (parent == desPortTree[grandpa].rightChild)) {
				desPortTree[parent].color = BLACK;
				desPortTree[grandpa].color = RED;
				//left_rotate(grandpa);
				int x = grandpa;
				int y = desPortTree[x].rightChild;
				desPortTree[x].rightChild = desPortTree[y].leftChild;
				if (desPortTree[y].leftChild != 0)
					desPortTree[desPortTree[y].leftChild].parent = x;
				desPortTree[y].parent = desPortTree[x].parent;
				if (*rootDes == x)
					*rootDes = y;
				else if (x == desPortTree[desPortTree[x].parent].leftChild)
					desPortTree[desPortTree[x].parent].leftChild = y;
				else
					desPortTree[desPortTree[x].parent].rightChild = y;
				desPortTree[y].leftChild = x;
				desPortTree[x].parent = y;
				//end of left_rotate
			}
		}

		desPortTree[*rootDes].color = BLACK;

	}//end of while
}

static void insert_des_tree_node(struct portTreeNode desPortTree[], struct fwRule rules[], int *desPortTreeCount, int i, int *rootDes)
{
	int parent;

	int idx = *rootDes;  //start from root, not 0 or 1
	int flag = 0;
	desPortTree[0].color = BLACK;

	//to find the right place and put the rule in the tree
	while(flag == 0) {
		//first: is root nil?
		if (*rootDes == 0) {
			//root is nil. put the rule in the root node.

			idx = (*desPortTreeCount)++;

			desPortTree[idx].color = BLACK;
			desPortTree[idx].startPort = rules[i].rule.desPortStart;
			desPortTree[idx].endPort = rules[i].rule.desPortEnd;

			if (i < 32) {
				unsigned int tmp = 1 << (31 - i);
				desPortTree[idx].matchRules[0] = desPortTree[idx].matchRules[0] | tmp;
			} else if (i < 64) {
				unsigned int tmp = 1 << (63 - i);
				desPortTree[idx].matchRules[1] = desPortTree[idx].matchRules[1] | tmp;
			} else if (i < 96) {
				unsigned int tmp = 1 << (95 - i);
				desPortTree[idx].matchRules[2] = desPortTree[idx].matchRules[2] | tmp;
			} else {
				unsigned int tmp = 1 << (127 - i);
				desPortTree[idx].matchRules[3] = desPortTree[idx].matchRules[3] | tmp;
			}

			desPortTree[idx].max = rules[i].rule.desPortEnd;

			*rootDes = idx;
			flag = 2;
		} else {
			//root is not nil.
			if (rules[i].rule.desPortStart < desPortTree[idx].startPort) {
				//packet's < node's: go to left child
				parent = idx;
				idx = desPortTree[idx].leftChild;

				if (idx == 0) {
					//we find what we need. new a idx node.

					idx = (*desPortTreeCount)++;

					desPortTree[idx].color = RED;
					desPortTree[idx].startPort = rules[i].rule.desPortStart;
					desPortTree[idx].endPort = rules[i].rule.desPortEnd;

					if (i < 32) {
						unsigned int tmp = 1 << (31 - i);
						desPortTree[idx].matchRules[0] = desPortTree[idx].matchRules[0] | tmp;
					} else if (i < 64) {
						unsigned int tmp = 1 << (63 - i);
						desPortTree[idx].matchRules[1] = desPortTree[idx].matchRules[1] | tmp;
					} else if (i < 96) {
						unsigned int tmp = 1 << (95 - i);
						desPortTree[idx].matchRules[2] = desPortTree[idx].matchRules[2] | tmp;
					} else {
						unsigned int tmp = 1 << (127 - i);
						desPortTree[idx].matchRules[3] = desPortTree[idx].matchRules[3] | tmp;
					}

					desPortTree[idx].max = rules[i].rule.desPortEnd;
					desPortTree[idx].parent = parent;

					desPortTree[parent].leftChild = idx;

					flag = 1;
				}
			} else if (rules[i].rule.desPortStart > desPortTree[idx].startPort) {
				parent = idx;
				idx = desPortTree[idx].rightChild;

				if (idx == 0) {
					idx = (*desPortTreeCount)++;

					desPortTree[idx].color = RED;
					desPortTree[idx].startPort = rules[i].rule.desPortStart;
					desPortTree[idx].endPort = rules[i].rule.desPortEnd;

					if (i < 32) {
						unsigned int tmp = 1 << (31 - i);
						desPortTree[idx].matchRules[0] = desPortTree[idx].matchRules[0] | tmp;
					} else if (i < 64) {
						unsigned int tmp = 1 << (63 - i);
						desPortTree[idx].matchRules[1] = desPortTree[idx].matchRules[1] | tmp;
					} else if (i < 96) {
						unsigned int tmp = 1 << (95 - i);
						desPortTree[idx].matchRules[2] = desPortTree[idx].matchRules[2] | tmp;
					} else {
						unsigned int tmp = 1 << (127 - i);
						desPortTree[idx].matchRules[3] = desPortTree[idx].matchRules[3] | tmp;
					}

					desPortTree[idx].max = rules[i].rule.desPortEnd;
					desPortTree[idx].parent = parent;

					desPortTree[parent].rightChild = idx;

					flag = 1;
				}
			} else if (rules[i].rule.desPortStart == desPortTree[idx].startPort) {
				if (rules[i].rule.desPortEnd == desPortTree[idx].endPort) {

					if (i < 32) {
						unsigned int tmp = 1 << (31 - i);
						desPortTree[idx].matchRules[0] = desPortTree[idx].matchRules[0] | tmp;
					} else if (i < 64) {
						unsigned int tmp = 1 << (63 - i);
						desPortTree[idx].matchRules[1] = desPortTree[idx].matchRules[1] | tmp;
					} else if (i < 96) {
						unsigned int tmp = 1 << (95 - i);
						desPortTree[idx].matchRules[2] = desPortTree[idx].matchRules[2] | tmp;
					} else {
						unsigned int tmp = 1 << (127 - i);
						desPortTree[idx].matchRules[3] = desPortTree[idx].matchRules[3] | tmp;
					}
					flag = 2;
					//we dont need a new node. so it is no need to color.
					//no need to deal with max.
				} else if (rules[i].rule.desPortEnd < desPortTree[idx].endPort) {
					parent = idx;
					idx = desPortTree[idx].leftChild;

					if (idx == 0) {
						//we find what we need. new a idx node.
						idx = (*desPortTreeCount)++;

						desPortTree[idx].color = RED;
						desPortTree[idx].startPort = rules[i].rule.desPortStart;
						desPortTree[idx].endPort = rules[i].rule.desPortEnd;

						if (i < 32) {
							unsigned int tmp = 1 << (31 - i);
							desPortTree[idx].matchRules[0] = desPortTree[idx].matchRules[0] | tmp;
						} else if (i < 64) {
							unsigned int tmp = 1 << (63 - i);
							desPortTree[idx].matchRules[1] = desPortTree[idx].matchRules[1] | tmp;
						} else if (i < 96) {
							unsigned int tmp = 1 << (95 - i);
							desPortTree[idx].matchRules[2] = desPortTree[idx].matchRules[2] | tmp;
						} else {
							unsigned int tmp = 1 << (127 - i);
							desPortTree[idx].matchRules[3] = desPortTree[idx].matchRules[3] | tmp;
						}

						desPortTree[idx].max = rules[i].rule.desPortEnd;
						desPortTree[idx].parent = parent;

						desPortTree[parent].leftChild = idx;

						flag = 1;
					}
				} else if (rules[i].rule.desPortEnd > desPortTree[idx].endPort) {
					parent = idx;
					idx = desPortTree[idx].rightChild;

					if (idx == 0) {
						idx = (*desPortTreeCount)++;

						desPortTree[idx].color = RED;
						desPortTree[idx].startPort = rules[i].rule.desPortStart;
						desPortTree[idx].endPort = rules[i].rule.desPortEnd;

						if (i < 32) {
							unsigned int tmp = 1 << (31 - i);
							desPortTree[idx].matchRules[0] = desPortTree[idx].matchRules[0] | tmp;
						} else if (i < 64) {
							unsigned int tmp = 1 << (63 - i);
							desPortTree[idx].matchRules[1] = desPortTree[idx].matchRules[1] | tmp;
						} else if (i < 96) {
							unsigned int tmp = 1 << (95 - i);
							desPortTree[idx].matchRules[2] = desPortTree[idx].matchRules[2] | tmp;
						} else {
							unsigned int tmp = 1 << (127 - i);
							desPortTree[idx].matchRules[3] = desPortTree[idx].matchRules[3] | tmp;
						}
						desPortTree[idx].max = rules[i].rule.desPortEnd;
						desPortTree[idx].parent = parent;

						desPortTree[parent].rightChild = idx;

						flag = 1;
					}
				}
			}
		}
	}
	//end of insertion: without coloring. no bug(Apr.28)

	if (flag != 2) {
		color_des_tree(desPortTree, rootDes, idx);
	}
}

static void calculate_src_tree_max(struct portTreeNode srcPortTree[], int rootSrc, int srcPortTreeCount)
{
	int i;
	for (i = 1; i < srcPortTreeCount; i++) {
		//from 1. 0 is nil.
		if ((srcPortTree[i].leftChild == 0) && (srcPortTree[i].rightChild == 0)) {
			int idx = i;

			while (idx != rootSrc) {
				if (srcPortTree[srcPortTree[idx].parent].max < srcPortTree[idx].max) {
					srcPortTree[srcPortTree[idx].parent].max = srcPortTree[idx].max;    
				}

				idx = srcPortTree[idx].parent;
			}
		}
	}

	srcPortTree[0].endPort = rootSrc;
}

static void calculate_des_tree_max(struct portTreeNode desPortTree[], int rootDes, int desPortTreeCount)
{
	int i;
	for (i = 1; i < desPortTreeCount; i++) {
		//from 1. 0 is nil.
		if ((desPortTree[i].leftChild == 0) && (desPortTree[i].rightChild == 0)) {
			int idx = i;

			while (idx != rootDes) {
				if (desPortTree[desPortTree[idx].parent].max < desPortTree[idx].max) {
					desPortTree[desPortTree[idx].parent].max = desPortTree[idx].max;    
				}

				idx = desPortTree[idx].parent;
			}
		}
	}

	desPortTree[0].endPort = rootDes;
}

static void firewall_rule_copy_to_gpu(int nf)
{
	if (nf == 0) {
		cudaMalloc((void**)&dev_srcAddrTrie, SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
		cudaMalloc((void**)&dev_desAddrTrie, DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
		cudaMalloc((void**)&dev_protocolHash, PROTOCOL_HASH_SIZE * 4 * sizeof(int));
		cudaMalloc((void**)&dev_srcPortTree, SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode));
		cudaMalloc((void**)&dev_desPortTree, DES_PORT_TREE_SIZE * sizeof(struct portTreeNode));

		cudaMemcpy(dev_srcAddrTrie, srcAddrTrie, SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_desAddrTrie, desAddrTrie, DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_protocolHash, protocolHash, PROTOCOL_HASH_SIZE * 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_srcPortTree, srcPortTree, SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_desPortTree, desPortTree, DES_PORT_TREE_SIZE * sizeof(struct portTreeNode), cudaMemcpyHostToDevice);
	} else {
		; /* Do nothing, let the NF do it */
	}
}

static void init(void)
{
	int i;

	protocolHash = (unsigned int *)malloc(PROTOCOL_HASH_SIZE * 4 * sizeof(unsigned int));
	for (i = 0; i < PROTOCOL_HASH_SIZE * 4; i++) {
		protocolHash[i] = 0;
	}

	srcAddrTrie = (struct trieAddrNode *)malloc(SRC_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	for (i = 0; i < SRC_ADDR_TRIE_SIZE; i++) {
		srcAddrTrie[i].matchRules[0] = 0;
		srcAddrTrie[i].matchRules[1] = 0;
		srcAddrTrie[i].matchRules[2] = 0;
		srcAddrTrie[i].matchRules[3] = 0;
		srcAddrTrie[i].rightChild = 0;
		srcAddrTrie[i].leftChild = 0;
	}

	desAddrTrie = (struct trieAddrNode *)malloc(DES_ADDR_TRIE_SIZE * sizeof(struct trieAddrNode));
	for (i = 0; i < DES_ADDR_TRIE_SIZE; i++) {
		desAddrTrie[i].matchRules[0] = 0;
		desAddrTrie[i].matchRules[1] = 0;
		desAddrTrie[i].matchRules[2] = 0;
		desAddrTrie[i].matchRules[3] = 0;
		desAddrTrie[i].rightChild = 0;
		desAddrTrie[i].leftChild = 0;
	}

	srcPortTree = (struct portTreeNode *)malloc(SRC_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	for (i = 0; i < SRC_PORT_TREE_SIZE; i++) {
		srcPortTree[i].parent = 0;
		srcPortTree[i].leftChild = 0;
		srcPortTree[i].rightChild = 0;
		srcPortTree[i].color = 0;
		srcPortTree[i].startPort = 0;
		srcPortTree[i].endPort = 0;
		srcPortTree[i].max = 0;
		srcPortTree[i].matchRules[0] = 0;
		srcPortTree[i].matchRules[1] = 0;
		srcPortTree[i].matchRules[2] = 0;
		srcPortTree[i].matchRules[3] = 0;
	}

	desPortTree = (struct portTreeNode *)malloc(DES_PORT_TREE_SIZE * sizeof(struct portTreeNode));
	for (i = 0; i < DES_PORT_TREE_SIZE; i++) {
		desPortTree[i].parent = 0;
		desPortTree[i].leftChild = 0;
		desPortTree[i].rightChild = 0;
		desPortTree[i].color = 0;
		desPortTree[i].startPort = 0;
		desPortTree[i].endPort = 0;
		desPortTree[i].max = 0;
		desPortTree[i].matchRules[0] = 0;
		desPortTree[i].matchRules[1] = 0;
		desPortTree[i].matchRules[2] = 0;
		desPortTree[i].matchRules[3] = 0;
	}
	desPortTree[0].startPort = 3;
}

void firewall_rule_construct(struct fwRule *rules, int rule_num, int nf)
{
	int rootSrc = 0;
	int rootDes = 0;

	int srcPortTreeCount = 0;
	int desPortTreeCount = 0;
	int srcAddrTrieCount = 0;
	int desAddrTrieCount = 0;

	int i;

	init();

	for (i = 0; i < rule_num; i++)
	{
		build_src_trie_node(rules, srcAddrTrie, &srcAddrTrieCount, i);

		unsigned int tmp = 1 << (31 - i);

		//rules[i].rule.desPortStart = 1234;
		//rules[i].rule.desPortEnd = 1234;

		build_des_trie_node(rules, desAddrTrie, &desAddrTrieCount, i);

		//printf("i %d pro %d\n", i, rules[i].rule.protocol);

		rules[i].rule.protocol = TYPE_TCP;

		if (i < 32) {
			protocolHash[rules[i].rule.protocol] = protocolHash[rules[i].rule.protocol] | tmp;
		} else if (i < 64) {
			protocolHash[rules[i].rule.protocol+4] = protocolHash[rules[i].rule.protocol+4] | tmp;
		} else if (i < 96) {
			protocolHash[rules[i].rule.protocol+8] = protocolHash[rules[i].rule.protocol+8] | tmp;
		} else {
			protocolHash[rules[i].rule.protocol+12] = protocolHash[rules[i].rule.protocol+12] | tmp;
		}

		insert_src_tree_node(srcPortTree, rules, &srcPortTreeCount, i, &rootSrc);

		insert_des_tree_node(desPortTree, rules, &desPortTreeCount, i, &rootDes);

	}//end of for in all the rules

	calculate_src_tree_max(srcPortTree, rootSrc, srcPortTreeCount);
	calculate_des_tree_max(desPortTree, rootDes, desPortTreeCount);

	firewall_rule_copy_to_gpu(nf);
}
