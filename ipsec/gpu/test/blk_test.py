#!/usr/bin/python
import os


pkt_size= 1024
for blk_num in range(1, 25, 1):
    pkt_num = 128 * blk_num
    cmd = './run ' + str(pkt_num) + " " + str(pkt_size) + ' ' +  str(blk_num)
    os.system(cmd)
