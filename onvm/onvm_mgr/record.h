#ifndef _record_H_
#define _record_H_
#include<stdio.h>
#define RECORD_CNT 300
FILE* open_record_file(char*);
void close_record_file(FILE*);

void record_tput_data(double rx_gbps,double tx_gbps);
void record_latency_data(int i,char* nf,double pre_lt,double h2d_lt,double kernel_lt,double d2h_lt,double post_lt,
                        double gpu_overhead,double cpu_overhead);

#endif // !_record_H_
