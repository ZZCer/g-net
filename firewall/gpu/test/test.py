#!/usr/bin/python
import os

pkt_num = 20000
for sm_num in range(1, 28, 1):
    cmd = './run ' + str(pkt_num) + ' ' + str(sm_num)
    os.system(cmd)
