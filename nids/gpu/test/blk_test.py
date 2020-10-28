#!/usr/bin/python
import os

pkt_len=64
for blk_num in range(1, 25, 1):
    pkt_num = 1000 * blk_num
    cmd = './run ' + str(pkt_num) + ' ' + str(pkt_len) + ' ' + str(blk_num)
    os.system(cmd)
