#ifndef _synchronize_H_
#define _synchronize_H_
#include <stdint.h>

//目前的同步标记大小srcip dstip srcport dstport proto
//整个系统的工作目录在onvm
#define HINT_SIZE 8
#define NFV_JSON "./NFVs.json"
//C程序的细节
uint16_t get_sync_plan(void);
int load_nfv_json(int* service_chain);

#endif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 