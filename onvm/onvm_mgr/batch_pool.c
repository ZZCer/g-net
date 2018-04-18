#include <rte_mempool.h>
#include <onvm_common.h>
#include <onvm_framework.h>
#include "batch_pool.h"

static void batch_pool_init(struct rte_mempool *mp, void *opaque) {
    UNUSED(mp);
    UNUSED(opaque);
}

static void batch_init(struct rte_mempool *mp,
		void *opaque, void *obj, unsigned cnt) {
    UNUSED(mp);
    UNUSED(opaque);
    UNUSED(cnt);
    new_batch_t *batch = (new_batch_t *)obj;
    memset(batch, 0, sizeof(new_batch_t));
}

int init_batch_pool(void) {
    rte_mempool_create(BATCH_POOL_NAME, BATCH_POOL_SIZE, sizeof(new_batch_t),
        BATCH_CACHE_SIZE, 0, batch_pool_init, NULL, batch_init, NULL, rte_socket_id(), 0);
    return 0;
}
