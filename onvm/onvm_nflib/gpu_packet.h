#ifndef __GPU_PACKET_H__
#define __GPU_PACKET_H__

#include <inttypes.h>

#define GPU_MAX_PKT_LEN 1600
#define GPU_PKT_ALIGN 16

//Synchronization Global variable
#define SYNC_DATA_COUNT 5
#define SYNC_DATA_SIZE 16

//GPU同步数据类型
#define SYNC_SOURCE_IP   0
#define SYNC_DEST_IP     1
#define SYNC_SOURCE_PORT 2
#define SYNC_DEST_PORT   3
#define SYNC_TCP_FLAGS   4

typedef struct gpu_packet_s {
	uint8_t  proto_id;
	uint8_t  tcp_flags;     /**< TCP flags if present */
	uint16_t payload_size;
	uint32_t src_addr;              /**< source address */
	uint32_t dst_addr;              /**< destination address */
	uint16_t src_port;      /**< source port. */
	uint16_t dst_port;      /**< destination port. */
	uint32_t sent_seq;      /**< TCP TX data sequence number. */
	uint32_t recv_ack;      /**< TCP RX data acknowledgement sequence number. */
	uint8_t  __padding[8];
	uint8_t  payload[0];
} __attribute__((aligned(16))) gpu_packet_t;

#endif // __GPU_PACKET_H__