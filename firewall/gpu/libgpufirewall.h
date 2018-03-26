#ifndef LIBGPUFIREWALL_H
#define LIBGPUFIREWALL_H

void firewall_kernel( struct pcktFive *dev_pcktFwFives, unsigned int *dev_res, int pcktCount, int block_num, int threads_per_blk);
void firewall_rule_construct(struct fwRule *rules, int rule_num, int nf);

#endif
