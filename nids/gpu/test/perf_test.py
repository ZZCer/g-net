#!/usr/bin/python
import os

pkt_len=1024-46
sm_num=28
#for pkt_num in range(128, 2049, 128):
for pkt_num in range(8192, 8192*28, 8192):
    cmd = './run ' + str(pkt_num) + ' ' + str(pkt_len) + ' ' + str(sm_num)
    os.system(cmd)
