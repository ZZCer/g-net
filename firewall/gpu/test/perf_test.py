#!/usr/bin/python
import os

blk=1
for pkt_num in range(200, 10000, 200):
        cmd = './run ' + str(pkt_num) + ' ' + str(blk)
        os.system(cmd)
