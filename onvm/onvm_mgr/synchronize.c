#include<rte_ring.h>
#include"synchronize.h"
#include"onvm_init.h"
#include"onvm_common.h"
#include"onvm_args.h"

//位于onvm_init的client数组
//没法直接通过nf_info_queue去获取nf的
extern uint16_t num_clients;
extern uint16_t plan[MAX_CLIENTS];
extern uint8_t last_plan;
extern struct client* clients;

uint16_t getSyncPlan(void){
    uint8_t LMASK=0xFF;
    uint16_t HMASK=0xFFFF;
    uint16_t dirty;
    uint16_t RV[MAX_CLIENTS];//GR|CR
    uint16_t WV[MAX_CLIENTS];//CW|GW
	uint16_t curPlan = 0;//这个current plan是用来表示最后加入nf服务链的client对应的plan

	int instance0=1;
	plan[instance0]=0;

	//nf的下标是从1开始的
	for (int i = 1; i <= num_clients; i++) {	
		struct client *cl=&clients[i];

		//h2d|d2h
		RV[i]=cl->hint.GR<<8|cl->hint.CR;
		WV[i]=cl->hint.CW<<8|cl->hint.GW;

		plan[instance0]=plan[instance0]|(RV[i]&HMASK)|(WV[i]<<8);
	}

	curPlan = plan[instance0];
	dirty = WV[instance0];

	//接下来，这里要从第2个NF开始
	for(int i=2;i<= num_clients;i++)
	{
		plan[i]=dirty&RV[i];
		dirty=(dirty^plan[i])|WV[i];

		if(i == num_clients)
			curPlan = plan[i];
	}

	//这个last plan是专门用来最后unload数据的
	last_plan=dirty & LMASK;    

	return curPlan;
}