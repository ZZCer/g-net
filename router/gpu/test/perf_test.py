#!/usr/bin/python
import os


for pkt_num in range(2000,100000, 2000):
    cmd = './run ' + str(pkt_num)
    os.system(cmd)
