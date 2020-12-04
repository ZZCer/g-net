#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <gpu_packet.h>

extern "C" __global__ void nat(gpu_packet_t **input_buf,const int job_num,
        const uint32_t *InternalIp,
        const uint32_t *ExternalIp,
        uint16_t* Port2Port,
        uint32_t* Port2Ip,
        char * PortSet, 
        const uint32_t *Mask,
        uint8_t* result,
        uint8_t* IPSet,
        
        //同步相关的参数
        uint16_t h2d_num,
        uint16_t d2h_num,
        const char* sync_in,
        char* sync_out,
        uint16_t* h2d_offset,
        uint16_t* d2h_offset
    ) 
{
        int id=threadIdx.x+blockIdx.x*blockDim.x;
        int step=blockDim.x*gridDim.x;
        int i = 0;
        int j = 0;
        
        if(h2d_num != 0)
        {
            //gpkt可以直接通过偏移量来进行数据同步
            char data[SYNC_DATA_SIZE];
            for(i = id ; i < job_num ; i += step){
                for(j = 0;j < SYNC_DATA_SIZE; j++)
                    data[j] = *(sync_in + i*SYNC_DATA_SIZE + j);
                
                for(j = 0;j < h2d_num ;j++)
                {
                    if(h2d_offset[j] <= 4)
                    {
                        if(h2d_offset[j] == 0)
                            input_buf[i]->src_addr = *((uint32_t*)(data + h2d_offset[j]));
                        else
                            input_buf[i]->dst_addr = *((uint32_t*)(data + h2d_offset[j]));
                    }
                    else if(h2d_offset[j] <= 10)
                    {
                        if(h2d_offset[j]==10)
                            input_buf[i]->dst_port = *((uint16_t*)(data + h2d_offset[j]));
                        else
                            input_buf[i]->src_port = *((uint16_t*)(data + h2d_offset[j]));
                    }
                }
            }
        }
        
        //这里采用的并行化思路是类似扫描的并行化思路
        for(int tid=id;tid<job_num;tid+=step)
        {
            uint32_t dstIP=input_buf[tid]->dst_addr;
            uint32_t srcIP=input_buf[tid]->src_addr;
            uint16_t dstPort=input_buf[tid]->dst_port;
            uint16_t srcPort=input_buf[tid]->src_port;

            result[tid]=1;

            //首先判断是否在同一个子网内部
            if(((*InternalIp) & (*Mask))==( srcIP & (*Mask)))
            {
                uint16_t src = 0;
                if(IPSet[(uint16_t)(srcIP & 0xFFFF)]!=0 && PortSet[srcPort]!=0)
                    src = IPSet[(uint16_t)(srcIP & 0xFFFF)];
                else{
                    while(PortSet[src]!=0)
                    {
                        src=(src+1)%65535;
                        if(src==0)
                            src=1;
                    }

                
                    //映射处理
                    PortSet[src]=1;
                    Port2Port[src]=srcPort;
                    Port2Ip[src]=srcIP;
                    IPSet[(uint16_t)(srcIP & 0xFFFF)]=src;
                }
                         
                input_buf[tid]->dst_addr=*(uint32_t*)ExternalIp;
                input_buf[tid]->dst_port=(uint16_t)0x00FF & src;

                //用这个result在后处理中判断要不要drop掉 1不drop 否则drop
                result[tid]=1;
            }
            else
            {
                //外网
                //被端口号过滤掉了
                if(PortSet[dstPort]==0 || dstIP==(*ExternalIp))
                    result[tid]=0;
                else
                {
                    uint16_t dst=Port2Port[dstPort];
                    uint32_t dstip0=Port2Ip[dstPort];

                    Port2Ip[dstPort]=0;
                    Port2Port[dstPort]=0;
                    PortSet[dstPort]=0;
                    IPSet[(uint16_t)(dstip0 & 0xFFFF)]=0;

                            
                    input_buf[tid]->src_addr=dstip0;
                    input_buf[tid]->src_port=dst; 

                    result[tid]=1;
                }
            }
        }
        
        if(d2h_num != 0)
        {
            char *data = NULL;
            for(i = id ; i < job_num ; i += step){
                //得到起点拷贝数组
                data = sync_out + i * SYNC_DATA_SIZE;	
                for(j = 0;j < d2h_num ;j++)
                {
                    if(d2h_offset[j] <= 4)
                        *((uint32_t*)(data + d2h_offset[j])) = ((d2h_offset[j] == 0) ? input_buf[i]->src_addr : input_buf[i]->dst_addr);
                    else if(h2d_offset[j] <= 10)
                        *((uint16_t*)(data + d2h_offset[j])) = ((d2h_offset[j] == 8) ? input_buf[i]->src_port : input_buf[i]->dst_port);
                }
            }
        }
}   
