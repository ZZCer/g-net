#!/usr/bin/python
import os


for pkt_size in [64, 128, 256, 512, 1024, 1500]:
	for pkt_num in [64, 128, 256, 512, 1024]:
		cmd = './run ' + str(pkt_num) + ' ' + str(pkt_size)
		os.system(cmd)
	print ''
