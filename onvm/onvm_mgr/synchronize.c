#include<rte_ring.h>
#include<rte_common.h>
#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<fcntl.h>
#include<cjson/cJSON.h>
#include"synchronize.h"
#include"onvm_init.h"
#include"onvm_common.h"
#include"onvm_args.h"

//位于onvm_init的client数组
//没法直接通过nf_info_queue去获取nf的
extern uint16_t num_clients;
extern uint16_t plan[MAX_CLIENTS];
extern uint8_t last_plan;

//从json文件中读取nf并且解析
extern char* global_service_chain[MAX_CLIENTS];
extern uint8_t CR_hints[MAX_CLIENTS]; 
extern uint8_t CW_hints[MAX_CLIENTS]; 
extern uint8_t GR_hints[MAX_CLIENTS]; 
extern uint8_t GW_hints[MAX_CLIENTS]; 

//初始化同步计划相关的函数也在里面
static uint8_t parse_str_to_binary(char* hint)
{
    uint8_t ht = 0;
    //字符串二进制转换

	//右移操作是补零的
    for(size_t i=0;i<HINT_SIZE && atoi(hint) != 0;i++) 
        ht |= ((hint[i]-'0') << (HINT_SIZE-i-1));

    return ht;
}

static void get_service_chain(int* service_chain)
{
	for(size_t i = 1 ; i <= (num_clients + 1) && service_chain!=NULL ; i++)
	{
		if(strcmp(global_service_chain[i],"NF_ROUTER") ==0 )
			service_chain[i-1] = NF_ROUTER;
		else if(strcmp(global_service_chain[i],"NF_NAT") == 0)
			service_chain[i-1] = NF_NAT; 
		else
			service_chain[i-1] = NF_END;
	}
}

int load_nfv_json(int* service_chain)
{
	FILE* fp = fopen(NFV_JSON,"r");
	if(fp == NULL)
	{
		rte_exit(EXIT_FAILURE,"No NFVs json file");
		return -1;
	}

	printf("Open Json successfully\n");

	char buffer[64];
	char* JSON_str = NULL;

	size_t size = 1;
	while( fgets(buffer,64,fp) != NULL )
	{
		size = sizeof(buffer);
		if(JSON_str == NULL)
			JSON_str = (char*)malloc(size);
		else
		{
			JSON_str = (char*)realloc(JSON_str,strlen(JSON_str)+size);
			//这一步取大小不能用sizeof，因为取出来的会是char*的大小为8字节
		}
		strcat(JSON_str,buffer);
	}
	
	//printf("\n%s\n",JSON_str);
    cJSON* json = cJSON_Parse(JSON_str);
    if(json == NULL)
    {
		rte_exit(EXIT_FAILURE,"Cant parse json\n");
        return -1;
    }

    const cJSON* NF_number = NULL;
    const cJSON* NFVs = NULL;
    const cJSON* NF = NULL;

	NF_number = cJSON_GetObjectItemCaseSensitive(json,"NFV_Number");
	if(NF_number == NULL)
	{	
		rte_exit(EXIT_FAILURE,"Cant get NFV number,probably not setting the value in json\n");
		return -1;
	}

	num_clients = atoi(NF_number->valuestring);
	printf("Number of nfvs is %d\n",num_clients);

	NFVs= cJSON_GetObjectItemCaseSensitive(json,"NFVs");
	if(NFVs == NULL)
	{	
		rte_exit(EXIT_FAILURE,"Cant get NFV Array,probably not setting the value in json\n");
		return -1;
	}

	int nfv_id = 1;
	cJSON_ArrayForEach(NF,NFVs)
	{
		//需要注意，C/C++多个指针是如何同时声明的
		const cJSON* name = NULL;
		const cJSON* CR = NULL , *CW = NULL;
		const cJSON* GR = NULL , *GW = NULL;

		name = cJSON_GetObjectItemCaseSensitive(NF,"Name");
		CR = cJSON_GetObjectItemCaseSensitive(NF,"CR_Hint");    
		GR = cJSON_GetObjectItemCaseSensitive(NF,"GR_Hint");    
		CW = cJSON_GetObjectItemCaseSensitive(NF,"CW_Hint");    
		GW = cJSON_GetObjectItemCaseSensitive(NF,"GW_Hint");    
	
		if(global_service_chain[nfv_id] == NULL)
			global_service_chain[nfv_id] = (char*)malloc(sizeof(name->valuestring));
		
		strcpy(global_service_chain[nfv_id],name->valuestring);
		CR_hints[nfv_id] = parse_str_to_binary(CR->valuestring);
		GR_hints[nfv_id] = parse_str_to_binary(GR->valuestring);
		CW_hints[nfv_id] = parse_str_to_binary(CW->valuestring);
		GW_hints[nfv_id] = parse_str_to_binary(GW->valuestring);

		nfv_id++;
	}

	printf("NF service chain\n");
	for(size_t i = 1 ; i <= (num_clients+1) ; i++)
	{
		printf("NF %ld is %s  CR:%d\n",i,global_service_chain[i],CR_hints[i]);
	}

	cJSON_Delete(json);

	get_service_chain(service_chain);

	return 1;
}

uint16_t get_sync_plan(void)
{
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
		//h2d|d2h
		RV[i]=GR_hints[i]<<8|CR_hints[i];
		WV[i]=CW_hints[i]<<8|GW_hints[i];

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

	for(int i=1;i<=num_clients;i++)
		printf("nf %d     plan %d\n",i,plan[i]);

	//这个last plan是专门用来最后unload数据的
	last_plan=dirty & LMASK;    

	printf("last plan:%d\n",last_plan);

	return curPlan;
}