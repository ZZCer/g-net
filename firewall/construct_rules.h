#include "gpu/firewall_kernel.h"

static void construct_rules(struct fwRule *rules)
{
	FILE * f = freopen("data.in", "r", stdin);
	if (f == NULL) {
		printf("file open error\n");
	}

	int i;
	struct inputItem *in = (struct inputItem *)malloc(RULESIZE * sizeof(struct inputItem));

	for (i = 0; i < RULESIZE; i ++) {
		in[i].srcAddr[0] = 0;
		in[i].srcAddr[1] = 0;
		in[i].srcAddr[2] = 0;
		in[i].srcAddr[3] = 0;
		in[i].srcMask = 0;
		in[i].desAddr[0] = 0;
		in[i].desAddr[1] = 0;
		in[i].desAddr[2] = 0;
		in[i].desAddr[3] = 0;
		in[i].desMask = 0;

		in[i].srcPort[0] = 0;
		in[i].srcPort[1] = 0;
		in[i].desPort[0] = 0;
		in[i].desPort[1] = 0;

		in[i].aChar[0] = '\0';
		in[i].aChar[1] = '\0';
		in[i].aChar[2] = '\0';
		in[i].aChar[3] = '\0';
		in[i].bChar[0] = '\0';
		in[i].bChar[1] = '\0';
		in[i].bChar[2] = '\0';
		in[i].bChar[3] = '\0';
		in[i].bChar[4] = '\0';
		in[i].bChar[5] = '\0';
		in[i].bChar[6] = '\0';
		in[i].bChar[7] = '\0';
	}

	for (i = 0; i < RULESIZE; i ++) {
		int f = scanf("@%d.%d.%d.%d/%d\t%d.%d.%d.%d/%d\t%d : %d\t%d : %d\t0x%c%c/0x%c%c\t0x%c%c%c%c/0x%c%c%c%c\n", 
				&in[i].srcAddr[0], &in[i].srcAddr[1], &in[i].srcAddr[2], &in[i].srcAddr[3], &in[i].srcMask,
				&in[i].desAddr[0], &in[i].desAddr[1], &in[i].desAddr[2], &in[i].desAddr[3], &in[i].desMask,
				&in[i].srcPort[0], &in[i].srcPort[1], &in[i].desPort[0], &in[i].desPort[1], 
				&in[i].aChar[0], &in[i].aChar[1], &in[i].aChar[2], &in[i].aChar[3], 
				&in[i].bChar[0], &in[i].bChar[1], &in[i].bChar[2], &in[i].bChar[3], 
				&in[i].bChar[4], &in[i].bChar[5], &in[i].bChar[6], &in[i].bChar[7]);
		if (f == 0) {
			printf("scanf error\n");
		}
	}

	for (i = 0; i < RULESIZE; i ++)
	{
		rules[i].rule.srcAddr = (in[i].srcAddr[0] << 24) | (in[i].srcAddr[1] << 16) | (in[i].srcAddr[2] << 8) | in[i].srcAddr[3];
		rules[i].rule.desAddr = (in[i].desAddr[0] << 24) | (in[i].desAddr[1] << 16) | (in[i].desAddr[2] << 8) | in[i].desAddr[3];
		rules[i].rule.srcMask = in[i].srcMask;
		rules[i].rule.desMask = in[i].desMask;
		rules[i].rule.srcPortStart = in[i].srcPort[0];
		rules[i].rule.srcPortEnd = in[i].srcPort[1];
		rules[i].rule.desPortStart = in[i].desPort[0];
		rules[i].rule.desPortEnd = in[i].desPort[1];

		rules[i].rule.protocol = rand() % 4;
	}

	firewall_rule_construct(rules, RULESIZE, 1);
}
