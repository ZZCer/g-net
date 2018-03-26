#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_ 

void scheduler_nf_spawn_new_thread(struct client *);
int scheduler_thread_main(void *);
void gpu_model_get_resource(int);

#endif
