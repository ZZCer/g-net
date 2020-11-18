#include"record.h"
#include<stdio.h>
#include<fcntl.h>
#include<stdlib.h>
#include<unistd.h>

FILE* stream = NULL;
char record_file[] = "../throughput.csv";
int count = 0;
int open_record_file(void)
{
    stream = fopen(record_file,"w+");
    if(stream != NULL)
    {
        char title[] = "rx,tx\n";
        fputs(title,stream);
        fflush(stream);
        return 1;
    }
    else
        return 0;  
}

void close_record_file(void)
{
    if(stream != NULL)
        fclose(stream);
}

void record_data(double rx_gbps,double tx_gbps)
{   
    if(count == 0)
        open_record_file();

    if(count < 300)
    {
        count++;

        char str[30];
        sprintf(str,"%.3f,%.3f\n",rx_gbps,tx_gbps);
        fputs(str,stream);
        if(count % 100==0)
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