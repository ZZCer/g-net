#include"record.h"
#include<stdio.h>
#include<fcntl.h>
#include<stdlib.h>
#include<math.h>
#include<unistd.h>
#include"onvm_common.h"

char record_tput_file[] = "../ThroughData/throughput.csv";
uint8_t init_flag = 0;

//注意行指针写法
static void init_latency_files(char (*files)[100])
{
    for(int i = 0 ; i < MAX_CLIENTS ; i++)
        strcpy(files[i],"../LatencyData/");
}

FILE* open_record_file(char* file)
{
    FILE* stream = NULL;
    stream = fopen(file,"w+");
    if(stream != NULL)
    {
        if(file == record_tput_file)
        {
            char title[] = "rx,tx\n";
            fputs(title,stream);
            fflush(stream);
            return stream;
        }
        else
        {
            char title[] = "preprocess_latency(us),HtoD_latency(us),kernel_latency(us),DtoH_latency(us),postprocess_latency(us),gpu_overhead(us),cpu_overhead\n";
            fputs(title,stream);
            fflush(stream);
            return stream;
        }
    }
    else
        return NULL;  
}

void close_record_file(FILE* stream)
{
    if(stream != NULL)
        fclose(stream);
}

void record_tput_data(double rx_gbps,double tx_gbps)
{   
    static FILE* stream = NULL;
    static int count = 0;

    if(count == 0 && stream == NULL)
        stream = open_record_file(record_tput_file);

    if(count > RECORD_CNT || stream == NULL)
        return ;

    if(count < RECORD_CNT)
    {
        count++;

        char str[30];
        sprintf(str,"%.3f,%.3f\n",rx_gbps,tx_gbps);
        fputs(str,stream);
        if(count % RECORD_CNT == 0)
            fflush(stream);
    }
    else 
    {
        if(stream != NULL)
        {
            fclose(stream);
            stream = NULL;
        }
    }
}


void record_latency_data(int i,char* nf,double pre_lt,double h2d_lt,double kernel_lt,double d2h_lt,double post_lt,
                        double gpu_overhead,double cpu_overhead)
{

    if(isnan(pre_lt) || isnan(h2d_lt) || isnan(kernel_lt) || isnan(d2h_lt) || isnan(post_lt))
        return ;

    static int count[MAX_CLIENTS];
    static FILE* stream[MAX_CLIENTS];
    static char files[MAX_CLIENTS][100];

    if(init_flag == 0)
    {   
        init_latency_files(files);
        init_flag = 1;
    }

    if(count[i] == 0 && stream[i] == NULL)
    {    
        strcat(files[i],nf);
        strcat(files[i],".csv");
        stream[i] = open_record_file(files[i]);
    }
    
    if(count[i] > RECORD_CNT || stream[i] == NULL)
        return ;

    if(count[i] < RECORD_CNT)
    {
        count[i]++;

        char str[100];
        sprintf(str,"%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",pre_lt,h2d_lt,kernel_lt,d2h_lt,post_lt,gpu_overhead,cpu_overhead);
        fputs(str,stream[i]);
        if(count[i] % RECORD_CNT == 0)
            fflush(stream[i]);
    }
    else 
    {
        if(stream[i] != NULL)
        {
            fclose(stream[i]);
            stream[i] = NULL;
        }
    }
}