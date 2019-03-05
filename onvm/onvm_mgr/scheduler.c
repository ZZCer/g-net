#include <stdint.h>
#include <math.h>
#include <rte_time.h>

#include "onvm_init.h"
#include "onvm_stats.h"
#include "onvm_nf.h"
#include "onvm_common.h"
#include "scheduler.h"

#define P_PERF (1.5)

#define MINB_BPS 256
#define MINB_PPS 1024

#define LATENCY_BOUND_BPS 100
#define LATENCY_BOUND_PPS 10
#define LATENCY_BOUND_OVERALL 100

//#define SET_RESOURCE 1

static int allocated_sm_num;
static int exausted;

void
scheduler_nf_spawn_new_thread(struct client *cl) {
	if (cl->gpu_info->launch_worker_thread != 0)
		rte_exit(EXIT_FAILURE, "Launch_worker_thread variable is not cleared\n");

	cl->gpu_info->launch_worker_thread = 1;
}

static void
gpu_get_resource(int instance_id, double T0) {
	struct client *cl = &(clients[instance_id]);
	double k1, b1, k2, b2;
	uint16_t L = cl->gpu_info->latency_us;
	unsigned int stream_num = cl->gpu_info->thread_num;

	printf("[%d] Current Resource Allocated: blk_num %d, batch_size %d, threads_per_blk %d\n",
			instance_id, cl->blk_num, cl->batch_size, cl->threads_per_blk);

	/* check N SMs from N = 1 */
	double L0, preL, L_bound, Lk;
	unsigned int N;
	int B0, minB;
	unsigned int i, success = 0;

	if (stream_num == 0 || T0 == 0 || cl->avg_pkt_len == 0) {
		/* Threads have not been initiated, return */
		RTE_LOG(ERR, APP, "stream num %d, T %.2lf, avg pkt len %d\n", stream_num, T0, cl->avg_pkt_len);
		return;	
	}

	if ((cl->gpu_info->para_num == 0) || (cl->gpu_info->para_num > MAX_PARA_NUM))
		rte_exit(EXIT_FAILURE, "Parameter not installed\n");

	/* Get the arguments */
	if (cl->nf_type == NF_BPS) {
		for (i = 0; i < cl->gpu_info->para_num; i ++) {
			if (abs(cl->gpu_info->pkt_size[i] - cl->avg_pkt_len) > 10)
				continue;

			k1 = cl->gpu_info->k1[i];
			b1 = cl->gpu_info->b1[i];
			k2 = cl->gpu_info->k2[i];
			b2 = cl->gpu_info->b2[i];
			RTE_LOG(INFO, APP, "[%d] Avg pkt size is %d, choose pkt size %d, k1 %.6f, b1 %.6f, k2 %.6f, b2 %.6f\n", 
					cl->instance_id, cl->avg_pkt_len, cl->gpu_info->pkt_size[i], k1, b1, k2, b2);
			break;
		}

		if (i == cl->gpu_info->para_num)
			RTE_LOG(ERR, APP, "No kernel parameters for the current packet length %d\n", cl->avg_pkt_len);
			//rte_exit(EXIT_FAILURE, "No kernel parameters for the current packet length %d\n", cl->avg_pkt_len);

		minB = MINB_BPS < cl->gpu_info->line_start_batch[i]? cl->gpu_info->line_start_batch[i] : MINB_BPS;
		L_bound = LATENCY_BOUND_BPS;
	} else if (cl->nf_type == NF_PPS) {
		k1 = cl->gpu_info->k1[0];
		b1 = cl->gpu_info->b1[0];
		k2 = cl->gpu_info->k2[0];
		b2 = cl->gpu_info->b2[0];
		minB = MINB_PPS;
		L_bound = LATENCY_BOUND_PPS;
	} else {
		rte_exit(EXIT_FAILURE, "NF Type Error, service id %d\n", cl->info->service_id);
	}

	if (L <= b1 + b2 + cl->cost_time) {
		printf("L = %d, b1 = %.lf, b2 = %lf, cost_time = %lf\n", L, b1, b2, cl->cost_time);
		printf(">>>>>>>>>>      Set latency is too low to meet the demand L <= b1 + b2 + cost_time      <<<<<<<<<<\n\n");
	}

	uint64_t batch_cnt = cl->stats.batch_cnt;
	if (batch_cnt == 0) batch_cnt = 1;
	/* Get the cost time */
	B0 = cl->stats.batch_size / batch_cnt;
	B0 = B0 / cl->blk_num;
	N = cl->blk_num * stream_num; /* all SMs allocated to the NF */
	if (B0 <= 0) {
		RTE_LOG(ERR, APP, "[%d] Batch size is 0, use the previous allocated resource in scheduling\n", instance_id);
		allocated_sm_num += N;
		return;
	}

	Lk = k1 * B0 + b1; /* kernel execution time */
	L0 = Lk + k2 * B0 * N + b2; /* expected overall gpu processing time */

	/* GPU time (kernel + PCIe + other costs) collected from clients */
	double measured_gpu_time = cl->stats.gpu_time / batch_cnt;
	/* Kernel execution time from Manager */
	double measured_kernel_time = cl->stats.kernel_time / cl->stats.kernel_cnt;
	/* Update cost time */
	//cl->cost_time = measured_gpu_time - L0;
	cl->cost_time = measured_gpu_time - L0 > cl->cost_time ? measured_gpu_time - L0 : cl->cost_time;

	printf("[%d] Batch size per blk is %d, total batch size is %d\n", cl->instance_id, B0, B0 * cl->blk_num);
	printf("[%d] Expected GPU time is %.2lf, Real GPU time is %.2lf\n", cl->instance_id, L0, measured_gpu_time);
	printf("[%d] Expected Kernel time is %.2lf, Real Kernel time is %.2lf\n", cl->instance_id, Lk, measured_kernel_time);
	printf("[%d] Expected Mem time is %.2lf\n", cl->instance_id, L0 - Lk);
	printf("[%d] Extra cost: %.2lf\n", cl->instance_id, cl->cost_time);

	/* Allocate resources */
	preL = 10000;
	for (N = stream_num; N < (unsigned int)SM_TOTAL_NUM - allocated_sm_num; N += stream_num) {
		/* First, calculate batch size with throughput */
		// following B0 is derived by T0 = B0*N/(L0+cost) = B0*N/(k1*B0+b1+k2*N*B0+b2 + cost)
		B0 = ceil(T0 * (b1 + b2 + cl->cost_time) / (N - T0 * (k1 + k2 * N)));
		if (B0 < 0)
			continue;
		if ((cl->nf_type == NF_BPS) && (B0 > MAX_THREAD_PER_BLK))
			continue;

		B0 = ((B0 % 64 == 0)? B0 : (B0/64 + 1) * 64);
		B0 = B0 < minB? minB : B0;

		/* Second, calculate the corresponding latency with the batch size, and check if it satisfy */
		L0 = k1 * B0 + b1 + k2 * B0 * N + b2 + cl->cost_time;

		if (L0 < L) {
			RTE_LOG(INFO, APP, "NF %d: Minimum SM number %d, Batch size is %d, Latency %.2lf us, T %.2lf Mpps\n", instance_id, N, B0, L0, T0);
			success = 1;
			break;
		} else if (preL - L0 < L_bound) {
			N -= stream_num;
			B0 = ceil(T0 * (b1 + b2 + cl->cost_time) / (N - T0 * (k1 + k2 * N)));
			B0 = B0 < minB? minB : B0;
			if (B0 <= MAX_THREAD_PER_BLK) {
				success = 1;
			}
			RTE_LOG(INFO, APP, "NF %d: Minimum SM number %d, Batch size is %d, Latency %.2lf us, T %.2lf Mpps, exit for limited latency benefit\n", instance_id, N, B0, L0, T0);
			break;
		}
		RTE_LOG(INFO, APP, "NF %d: Minimum SM number %d, Batch size per SM is %d, Latency = %.2lf us, T = %.2lf Mpps\n", instance_id, N, B0, L0, T0);
		preL = L0;
	}

	if (success == 1) {
		/* For each CPU thread: the least number of SMs to meet the latency demands */
		cl->blk_num = N / stream_num;
		/* The batch size for each CPU thread */
		cl->batch_size = B0 * cl->blk_num;// * batch_cnt; // TODO multiply by batch_cnt?
		cl->threads_per_blk = B0 < MAX_THREAD_PER_BLK ? B0 : MAX_THREAD_PER_BLK;

		allocated_sm_num += N;
	} else {
		RTE_LOG(ERR, APP, "\n>>>>>>>>>>      GPU resource is exausted      <<<<<<<<<<\n");

		/* Maintains the original resource allocation if there is available resources, or reallocate in the "if" */
		if (allocated_sm_num + cl->blk_num * stream_num > SM_TOTAL_NUM) {
			printf("Only allocate 1 SM for each thread\n\n");
			/* At least each kernel from a CPU thread should be allocated with one SM */
			if (allocated_sm_num + stream_num > SM_TOTAL_NUM) {
				rte_exit(EXIT_FAILURE, "GPU resource exausted even allocate one SM\n");
			}

			cl->blk_num = 1; //floor((SM_TOTAL_NUM - allocated_sm_num) / stream_num);
			if (cl->blk_num == 0) {
				printf("allocated SM is %d, the stream number of this NF is %d\n", allocated_sm_num, stream_num);
				cl->blk_num = 1;
			}
			cl->threads_per_blk = MAX_THREAD_PER_BLK;
			cl->batch_size = cl->threads_per_blk;
		} else {
			printf("Maintain previous resource allocation strategy\n\n");
		}

		allocated_sm_num += cl->blk_num * stream_num;
		exausted = 1;
	}


	RTE_LOG(INFO, APP, "Scheduling for NF %d: Throughput %lf Mpps, Latency %d us\n\t#SM %d, total allocated #SM %d, #SM per Stream %d, Batch size per CUDA stream (CPU thread) is %d, Batch size per SM is %d, threads_per_blk is %d\n\n",
			instance_id, cl->throughput_mpps, L,
			cl->blk_num * stream_num, allocated_sm_num, cl->blk_num, cl->batch_size, B0, cl->threads_per_blk);

	return;
}

static void
gpu_optimize_latency(double T0) {
	unsigned int i, id;
	int max_client_id = -1;
	double deltaL, max_deltaL;
	double k1, b1, k2, b2;
	struct client *cl;
	double newL, oldL;
	int B0, minB;
	unsigned int stream_num, N;
	
	printf("\n");
	RTE_LOG(INFO, APP, "Allocate SMs to further optimize the latency:\n");

	while (allocated_sm_num <= SM_TOTAL_NUM) {
		max_deltaL = 0;

		for (id = 0; id < MAX_CLIENTS; id ++) {
			cl = &(clients[id]);
			stream_num = cl->gpu_info->thread_num;

			if (!onvm_nf_is_valid(cl) || (cl->info->service_id == NF_PKTGEN)
					|| (cl->throughput_mpps == 0) || (stream_num == 0) || (cl->avg_pkt_len == 0))
				continue;

			if (allocated_sm_num + stream_num > SM_TOTAL_NUM)
				continue;

			if (cl->nf_type == NF_BPS) {
				for (i = 0; i < cl->gpu_info->para_num; i ++) {
					if (abs(cl->gpu_info->pkt_size[i] - cl->avg_pkt_len) > 10)
						continue;
					k1 = cl->gpu_info->k1[i];
					b1 = cl->gpu_info->b1[i];
					k2 = cl->gpu_info->k2[i];
					b2 = cl->gpu_info->b2[i];
					RTE_LOG(DEBUG, APP, "Avg pkt size is %d, choose pkt size %d, k1 %.2f, b1 %.2f, k2 %.2f, b2 %.2f\n", 
							cl->avg_pkt_len, cl->gpu_info->pkt_size[i], k1, b1, k2, b2);
					break;
				}
				minB = MINB_BPS < cl->gpu_info->line_start_batch[i]? cl->gpu_info->line_start_batch[i] : MINB_BPS;
			} else if (cl->nf_type == NF_PPS) {
				k1 = cl->gpu_info->k1[0];
				b1 = cl->gpu_info->b1[0];
				k2 = cl->gpu_info->k2[0];
				b2 = cl->gpu_info->b2[0];
				minB = MINB_PPS;
			} else {
				rte_exit(EXIT_FAILURE, "NF Type Error, service id %d\n", cl->info->service_id);
			}

			N = cl->blk_num * stream_num; /* total number of SMs to the NF */
			B0 = ceil(T0 * (b1 + b2 + cl->cost_time) / (N - T0 * (k1 + k2 * N)));
			if (B0 < 0)
				continue;
			if ((cl->nf_type == NF_BPS) && (B0 > MAX_THREAD_PER_BLK))
				continue;

			B0 = ((B0 % 64 == 0)? B0 : (B0/64 + 1) * 64);
			B0 = B0 < minB? minB : B0;
			oldL = k1 * B0 + b1 + k2 * N * B0 + b2 + cl->cost_time;

			/* Allocate one more SM */
			N = (cl->blk_num + 1) * stream_num;
			B0 = ceil(T0 * (b1 + b2 + cl->cost_time) / (N - T0 * (k1 + k2 * N)));
			if (B0 < 0)
				rte_exit(EXIT_FAILURE, "batch size < 0\n");
			if ((cl->nf_type == NF_BPS) && (B0 > MAX_THREAD_PER_BLK))
				rte_exit(EXIT_FAILURE, "batch size > MAX_THREAD_PER_BLK\n");

			B0 = ((B0 % 64 == 0)? B0 : (B0/64 + 1) * 64);
			B0  = B0 < minB? minB : B0;
			newL = k1 * B0 + b1 + k2 * N * B0 + b2 + cl->cost_time;

			deltaL = oldL - newL;
			if (max_deltaL < deltaL) {
				max_deltaL = deltaL;
				max_client_id = id;
				/* If one more SM to each kernel, then the total allocated extra SMs should
				 * equal to the CPU thread number */
			}
		}

		if (max_deltaL <= LATENCY_BOUND_OVERALL) {
			RTE_LOG(INFO, APP, "Limited latency improvement %.2lf, scheduling complete\n", max_deltaL);
			return;
		}

		cl = &(clients[max_client_id]);
		stream_num = cl->gpu_info->thread_num;

		if (cl->nf_type == NF_BPS) {
			for (i = 0; i < cl->gpu_info->para_num; i ++) {
				if (abs(cl->gpu_info->pkt_size[i] - cl->avg_pkt_len) > 10)
					continue;
				k1 = cl->gpu_info->k1[i];
				b1 = cl->gpu_info->b1[i];
				k2 = cl->gpu_info->k2[i];
				b2 = cl->gpu_info->b2[i];
				RTE_LOG(DEBUG, APP, "Avg pkt size is %d, choose pkt size %d, k1 %.2f, b1 %.2f, k2 %.2f, b2 %.2f\n", 
						cl->avg_pkt_len, cl->gpu_info->pkt_size[i], k1, b1, k2, b2);
				break;
			}
			minB = MINB_BPS < cl->gpu_info->line_start_batch[i]? cl->gpu_info->line_start_batch[i] : MINB_BPS;
		} else if (cl->nf_type == NF_PPS) {
			k1 = cl->gpu_info->k1[0];
			b1 = cl->gpu_info->b1[0];
			k2 = cl->gpu_info->k2[0];
			b2 = cl->gpu_info->b2[0];
			minB = MINB_PPS;
		} else {
			rte_exit(EXIT_FAILURE, "NF Type Error, service id %d\n", cl->info->service_id);
		}

		N = (cl->blk_num + 1) * stream_num;
		B0 = ceil(T0 * (b1 + b2 + cl->cost_time) / (N - T0 * (k1 + k2 * N)));
		if (B0 < 0)
			rte_exit(EXIT_FAILURE, "batch size < 0\n");
		if ((cl->nf_type == NF_BPS) && (B0 > MAX_THREAD_PER_BLK))
			rte_exit(EXIT_FAILURE, "batch size > MAX_THREAD_PER_BLK\n");

		B0 = ((B0 % 64 == 0)? B0 : (B0/64 + 1) * 64);
		B0  = B0 < minB? minB : B0;
		newL = k1 * B0 + b1 + k2 * N * B0 + b2 + cl->cost_time;

		/* Update the info */
		cl->blk_num ++;
		cl->batch_size = B0 * cl->blk_num;// * (cl->stats.batch_cnt <= 0 ? 1 : cl->stats.batch_cnt); // TODO multiply by batch_cnt?
		cl->threads_per_blk = B0 < MAX_THREAD_PER_BLK ? B0 : MAX_THREAD_PER_BLK;

		/* Allocate another stream_num SMs, one per stream */
		allocated_sm_num += stream_num;

		RTE_LOG(INFO, APP, "[allocated SM %d, available %d] choose NF %d, + %d SMs, Reduce latency %.2lf to %.2lf (w/ cost time), batch size per thread %d, batch size per SM %d, blk_num %d\n",
				allocated_sm_num, SM_TOTAL_NUM - allocated_sm_num, max_client_id, stream_num, max_deltaL, newL, cl->batch_size, B0, cl->blk_num);
	}

	RTE_LOG(INFO, APP, "---------- Scheduling Complete ----------\n\n");
}

static void
schedule(void) {
	unsigned int i;
	struct timespec end;

	clock_gettime(CLOCK_MONOTONIC, &end);

	/* Calculate the allocated SMs */
	allocated_sm_num = 0;
	exausted = 0;

	double minT = 10000;
	for (i = 0; i < MAX_CLIENTS; i ++) {
		if (!onvm_nf_is_valid(&clients[i]))
			continue;

		if (minT > clients[i].throughput_mpps)
			minT = clients[i].throughput_mpps;
	}

	/* Local Schedule */
	for (i = 0; i < MAX_CLIENTS; i ++) {
		if (!onvm_nf_is_valid(&clients[i]))
			continue;

		/* estimate resource allocation */
		gpu_get_resource(i, minT * P_PERF);
	}

	/* Global Schedule */
	if (exausted == 0)
		gpu_optimize_latency(minT * P_PERF);

	printf("\n========================================================\n\n");
}

#define STATE_SAMPLES 5
#define STEADY_STATE_D_THRESHOLD 20000

static int detect_steady_state(int timediff) {
	// We define that the system is in steady state if
	// whole system throughput is steady (samples have a low variance)
	static double samples[STATE_SAMPLES] = {0};
	static int current_index = 0;
	static uint64_t tx_last = 0;
	static int full = 0;

	uint64_t tx = rte_atomic64_read((rte_atomic64_t *)(uintptr_t)&ports->tx_stats.tx[0]); // FIXME: only support one port
	samples[current_index] = (tx - tx_last) / (double) timediff;
	tx_last = tx;

	if (current_index == STATE_SAMPLES - 1)
		full = 1;
	current_index = (current_index + 1) % STATE_SAMPLES;
	
	if (full) {
		double sum = 0.;
		for (int i = 0; i < STATE_SAMPLES; i++) {
			sum += samples[i];
		}
		double avg = sum / STATE_SAMPLES;
		double diff_square_sum = 0.;
		for (int i = 0; i < STATE_SAMPLES; i++) {
			diff_square_sum += (samples[i] - avg) * (samples[i] - avg);
		}
		double index_of_dispersion = diff_square_sum / STATE_SAMPLES / avg;
		
		printf("%f\n", index_of_dispersion);
		return index_of_dispersion < STEADY_STATE_D_THRESHOLD;
	} else return 0;
}

static void refresh_statistics(int timediff) {
	// Here we only check the performance stats of
	// 1. Throughput and latency of each NF
	// 2. Throughput of the entire system
	// TODO: Use built-in stats for now.
	UNUSED(timediff);
}

typedef enum schedule_resource_type_e { 
	BATCH_SIZE,
	NUM_WORKER,
	NUM_SM
} schedule_resource_type_t;

typedef struct schedule_decision_s {
	int instance_id;
	schedule_resource_type_t resource_type;
} schedule_decision_t;

static void
schedule_dynamic(int timediff) {
	int chain_ids[ONVM_MAX_CHAIN_LENGTH];

	// Make decisions only when in steady state
	// TODO: dirty implementation
	if (!detect_steady_state(timediff)) {
		RTE_LOG(INFO, APP, "Steady state not detected.\n");
		return;
	} else {
		allocated_sm_num = 0;
		for (int i = 0; i < MAX_CLIENTS; i++) {
			if (!onvm_nf_is_valid(&clients[i]))
				continue;
			chain_ids[clients[i].position_in_chain] = i;
			
			struct client* c = &clients[i];
			// c->blk_num = 4;
			allocated_sm_num += clients[i].blk_num * clients[i].worker_scale_target;
			RTE_LOG(INFO, APP, "Client %d: #SM - %d * %d, Batch Size - %d\n", c->instance_id, c->worker_scale_target, c->blk_num, c->batch_size);
		}
		
		for (int i = 1; i < default_chain->chain_length; i++) {
			struct client* c = &clients[chain_ids[i]];
			if (!onvm_nf_is_valid(c))
				continue;

			// check the performance metrics
			double latency = (c->stats.cpu_time + c->stats.gpu_time) / c->stats.batch_cnt;
			RTE_LOG(INFO, APP, "Client %d latency: %f, limit: %d\n", c->instance_id, latency, c->gpu_info->latency_us);
			if (latency > c->gpu_info->latency_us) {
				RTE_LOG(INFO, APP, "Client %d surpasses the latency limit.\n", c->instance_id);
				// latency optimization is required
				// c->batch_size *= 0.8; // todo
				if (c->worker_scale_target < 2) { // TODO thread limit
					unsigned int old_target = c->worker_scale_target;
					c->worker_scale_target += 1;
					allocated_sm_num -= c->blk_num * old_target;
					c->blk_num *= old_target / (double) c->worker_scale_target;
					if (c->blk_num == 0)
						c->blk_num = 1;
					allocated_sm_num += c->blk_num * c->worker_scale_target;
					c->batch_size *= old_target / (double) c->worker_scale_target;
				} else if (c->stats.kernel_time / c->stats.gpu_time > 0.4 && allocated_sm_num + c->worker_scale_target <= SM_TOTAL_NUM) {
					c->blk_num++;
					RTE_LOG(INFO, APP, "Client %d: Allocating %d SMs.\n", c->instance_id, c->worker_scale_target * c->blk_num);
				} else {
					c->batch_size *= 0.9;
				}
				return;
			} else {
				// check current throughput of NF
				if (i == 1) {
					continue;
				} else {
					if (clients[chain_ids[i - 1]].stats.tx_drop > 0) {
						// the current throughput of NF is low
						RTE_LOG(INFO, APP, "Increasing the performance of client %d\n", c->instance_id);
						// if kernel time takes the majority of the GPU time
						// we need to add SM to increase performance
						// if (c->stats.kernel_time / c->stats.gpu_time > 0.4) { // TODO: set threshold
						//  	c->blk_num++;
						// } else {
							c->batch_size *= 1.2;
						// }
						return;
					}
				}
			}
			// batch size limitation
			if (c->batch_size > MAX_BATCH_SIZE) {
				c->batch_size = MAX_BATCH_SIZE;
			}
		}
		// no scheduling decision yet, increase throughput of first nf.
		if (default_chain->chain_length >= 2) {
			// TODO: do not increase if the speed is approaching the input rate.
			RTE_LOG(INFO, APP, "Increasing the performance of head client of the service chain.\n");
			clients[chain_ids[1]].batch_size *= 1.1;
			if (clients[chain_ids[1]].batch_size > MAX_BATCH_SIZE) {
				clients[chain_ids[1]].batch_size = MAX_BATCH_SIZE;
			}
		}
	}

}

static void
schedule_static(void) {
	// unsigned int i;
	struct timespec end;

	clock_gettime(CLOCK_MONOTONIC, &end);

	/* Calculate the allocated SMs */
	allocated_sm_num = 0;
	exausted = 0;

	// TODO add new implementations
	/* Local Schedule */
	for (int i = 0; i < MAX_CLIENTS; i++) {
		if (!onvm_nf_is_valid(&clients[i]))
			continue;

		/* static allocation */
		struct client *cl = &(clients[i]);
		unsigned int stream_num = cl->gpu_info->thread_num;
		
		switch (cl->info->service_id) {
			case NF_ROUTER:
				cl->blk_num = 1;				// blk_num * stream_num <= total #SM	// 6.1 device max #SM: 28
				cl->batch_size = 4096; 		// max definition in onvm_common.h
				cl->threads_per_blk = 1024;		// 6.1 device max: 1024
				cl->worker_scale_target = 1;
				break;
			case NF_FIREWALL:
				cl->blk_num = 6;				// blk_num * stream_num <= total #SM	// 6.1 device max #SM: 28
				cl->batch_size = 2048; 		// max definition in onvm_common.h
				cl->threads_per_blk = 1024;		// 6.1 device max: 1024
				cl->worker_scale_target = 1;
				break;
			case NF_NIDS:
				cl->blk_num = 6;				// blk_num * stream_num <= total #SM	// 6.1 device max #SM: 28
				cl->batch_size = 2048; 		// max definition in onvm_common.h
				cl->threads_per_blk = 1024;		// 6.1 device max: 1024
				cl->worker_scale_target = 1;
				break;
			case NF_IPSEC:
				cl->blk_num = 6;				// blk_num * stream_num <= total #SM	// 6.1 device max #SM: 28
				cl->batch_size = 2048; 		// max definition in onvm_common.h
				cl->threads_per_blk = 1024;		// 6.1 device max: 1024
				break;
		}

		assert(cl->batch_size <= MAX_BATCH_SIZE);
		assert(stream_num * cl->blk_num <= SM_TOTAL_NUM);
	}

	printf("\n========================================================\n\n");
}

int
scheduler_thread_main(void *arg) {
	UNUSED(arg);
	const unsigned sleeptime = 1;

	RTE_LOG(INFO, APP, "Core %d: Scheduler is running\n", rte_lcore_id());

	while (1) {
		usleep(sleeptime * 1000000);
		onvm_nf_check_status();
		onvm_stats_display_all(sleeptime);
		// schedule();
		// schedule_dynamic(sleeptime);
		schedule_static();
		onvm_stats_clear_all_clients();
	}

	return 0;
}
