/*
 *  B-Queue -- An efficient and practical queueing for fast core-to-core
 *             communication
 *
 *  Copyright (C) 2011 Junchang Wang <junchang.wang@gmail.com>
 *
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef _FIFO_B_QUQUQ_H_
#define _FIFO_B_QUQUQ_H_

#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
#include <inttypes.h>
#include <string.h>
#include <stdint.h>
#include <rte_mbuf.h>

/* Enable the following macros will significantly degrade throughput */
//#define BACKTRACKING	1
//#define ADAPTIVE		1

typedef struct rte_mbuf * ELEMENT_TYPE;

#define QUEUE_SIZE (4096)
#define BQUEUE_BATCH_SIZE (256)
#define CONS_BATCH_SIZE BQUEUE_BATCH_SIZE
#define PROD_BATCH_SIZE BQUEUE_BATCH_SIZE
#define BATCH_INCREAMENT (BQUEUE_BATCH_SIZE/2)

#define CONGESTION_PENALTY (1000) /* cycles */

struct queue_t {
	/* Mostly accessed by producer. */
	volatile uint32_t head;
	volatile uint32_t batch_head;

	/* Mostly accessed by consumer. */
	volatile uint32_t tail __attribute__ ((aligned(64)));
	volatile uint32_t batch_tail;
	unsigned long batch_history;

	/* readonly data */
	uint64_t start_c __attribute__ ((aligned(64)));
	uint64_t stop_c;

	/* accessed by both producer and comsumer */
	ELEMENT_TYPE data[QUEUE_SIZE] __attribute__ ((aligned(64)));
} __attribute__ ((aligned(64)));

#define SUCCESS 0
#define BUFFER_FULL -1
#define BUFFER_EMPTY -2

void bq_queue_init(struct queue_t *q);
int bq_enqueue(struct queue_t *q, ELEMENT_TYPE value);
int bq_dequeue(struct queue_t *q, ELEMENT_TYPE *value);

#endif
