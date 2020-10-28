#!/usr/bin/python
import os


for pkt_num in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
	cmd = './run ' + str(pkt_num)
	os.system(cmd)
