#ifndef PSTACK_H
#define PSTACK_H

#include "parallel.h"

typedef struct pstack_thread_info_s {
    IP_THREAD_LOCAL_P ip_thread_local;
    TCP_THREAD_LOCAL_P tcp_thread_local;
} pstack_thread_info;

void pstack_init(pstack_thread_info info, int num_threads);

void pstack_process(char *data, int len, int rx_queue_id);

#endif // PSTACK_H