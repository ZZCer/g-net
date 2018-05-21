#ifndef FIREWALL_KERNEL_H
#define FIREWALL_KERNEL_H

#define RULESIZE 125

#define SRC_ADDR_TRIE_SIZE 1000
#define DES_ADDR_TRIE_SIZE 1000
#define PROTOCOL_HASH_SIZE 4
#define SRC_PORT_TREE_SIZE 1000
#define DES_PORT_TREE_SIZE 1000

#define BLACK 2
#define RED 1
#define WHITE 0

#define ACCEPT 1
#define REJECT -1

#define TYPE_TCP 1
#define TYPE_UDP 2
#define TYPE_ICMP 3
#define TYPE_DEFAULT 4

struct inputItem
{
    int srcAddr[4];
    int srcMask;

    int desAddr[4];
    int desMask;

    int srcPort[2];
    int desPort[2];

    char aChar[4];
    char bChar[8];
};

struct portTreeNode{
    int parent;
    int leftChild;
    int rightChild;
    int color;  //0-nil 1-red 2-black
    unsigned int matchRules[4];
    unsigned int startPort;
    unsigned int endPort;
    unsigned int max;
};  //src & des use the same struct

struct fwFive {
    unsigned int srcAddr;
    unsigned int desAddr;  
    int srcMask;
    int desMask;
    unsigned int srcPortStart;
    unsigned int srcPortEnd;
    unsigned int desPortStart;
    unsigned int desPortEnd;
    int protocol;
};

struct fwRule {
    struct fwFive rule;
    int order;
    int action;
};

struct trieAddrNode {
    unsigned int matchRules[4];
    int leftChild;
    int rightChild;
};  //src & des use the same struct

void firewall_rule_construct(struct fwRule *rules, int rule_num, int nf);

#endif
