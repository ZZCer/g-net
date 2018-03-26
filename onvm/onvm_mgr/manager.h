#ifndef _MANAGER_H_
#define _MANAGER_H_

struct callback_arg {
	uint16_t instance_id;
	uint16_t blk_num;
	uint32_t not_use;
};

typedef union {
	uint64_t arg;
	struct callback_arg info;
} type_arg;

int manager_thread_main(void *arg);
void init_manager(void);
void manager_nf_init(int);

#endif
