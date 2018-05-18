#ifndef _MANAGER_H_
#define _MANAGER_H_

#include <cuda.h>
#include "drvapi_error_string.h"

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

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
static inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
	if (CUDA_SUCCESS != err) {
		fprintf(stderr, "CUDA Driver API error = %04d  \"%s\" from file <%s>, line %i.\n",
				err, getCudaDrvErrorString(err), file, line );
		exit(-1);
	}
}

#endif
