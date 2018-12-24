#include "nids.h"
#include "onvm_common.h"
#include "pstack.h"

void pstack_init(pstack_thread_info info, int num_threads) {
    nids_lib_init(num_threads, info.ip_thread_local, info.tcp_thread_local);
}

inline void pstack_process(char *data, int len, int rx_queue_id) {
    gen_ip_frag_proc(data, len, rx_queue_id);
}

