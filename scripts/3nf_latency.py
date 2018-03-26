#!/usr/bin/python
import sys

sample_rate = 50

trace_file1 = "3nf_firewall"
trace_file2 = "3nf_ipsec"
trace_file3 = "3nf_nids"

trace1 = open(trace_file1, "r").readlines()
trace2 = open(trace_file2, "r").readlines()
trace3 = open(trace_file3, "r").readlines()

len1 = len(trace1)
len2 = len(trace2)
len3 = len(trace3)

minlen = min(len1, len2, len3)

total_latency = []

for i in range(0, minlen):
	total_latency.append((float)(trace1[i]) + (float)(trace2[i]) + (float)(trace3[i]))

total_latency.sort()

sample_interval = (int) (len(total_latency) / sample_rate)
sample_array = []

for i in range(0, minlen, sample_interval):
	sample_array.append(total_latency[i])
sample_array.append(total_latency[minlen-2])

for i in range(0, len(sample_array)):
	print sample_array[i]
