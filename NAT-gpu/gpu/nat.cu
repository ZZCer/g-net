#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <gpu_packet.h>

extern "C" __global__ void nat(gpu_packet_t **input_buf,const int job_num,
    const uint32_t *InternalIp,const uint32_t *ExternalIp,
    uint16_t* Port2Port,uint32_t* Port2Ip,char * PortSet, const uint32_t *Mask,uint8_t* result) {

        int id=threadIdx.x+blockIdx.x*blockDim.x;
        int step=blockDim.x*gridDim.x;

        //这里采用的并行化思路是类似扫描的并行化思路
        for(int tid=id;tid<job_num;tid+=step)
        {
            uint32_t dstIP=input_buf[tid]->dst_addr;
            uint32_t srcIP=input_buf[tid]->src_addr;
            uint16_t dstPort=input_buf[tid]->dst_port;
            uint16_t srcPort=input_buf[tid]->src_port;

            //首先判断是否在同一个子网内部
            if(((*ExternalIp) & (*Mask))==( srcIP & (*Mask)))
            {
                //内网
                uint16_t src=srcPort;
                while(PortSet[src]!=0)
                    src=(src+1)%65535;
                
                //映射处理
                PortSet[src]=1;
                Port2Port[src]=srcPort;
                Port2Ip[src]=srcIP;
                
                result[0]=*(uint32_t*)ExternalIp;
                result[4]=0x00FF & src;
            }
            else
            {
                //外网
                //被端口号过滤掉了
                if(PortSet[dstPort]==0)
                    result[4]=0;
                else if(dstIP==(*ExternalIp))//目标不是当前host ip
                    result[4]=0;
                else
                {
                    uint32_t dst=Port2Port[dstPort];
                    uint16_t dstip0=Port2Ip[dstPort];

                    Port2Ip[dstPort]=0;
                    Port2Port[dstPort]=0;

                    result[0]=*(uint32_t*)dstip0;
                    result[4]=0x00FFd & dst;
                }
            }
        }
}   
