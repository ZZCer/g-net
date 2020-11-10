#ifndef _record_H_
#define _record_H_
int open_record_file(void);
void close_record_file(void);

void record_data(double rx_gbps,double tx_gbps);

#endif // !_record_H_
