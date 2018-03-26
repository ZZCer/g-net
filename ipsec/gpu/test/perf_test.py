#!/usr/bin/python
import os

pkt_size=64-46
blk_num=1
for pkt_num in range(128, 8192, 128):
    cmd = './run ' + str(pkt_num) + ' ' + str(pkt_size) + ' ' + str(blk_num)
    os.system(cmd)
