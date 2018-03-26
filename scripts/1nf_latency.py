#!/usr/bin/python
import sys

sample_rate = 50

trace_file1 = "1nf_nids"

trace1 = open(trace_file1, "r").readlines()

len1 = len(trace1)

total_latency = []

for i in range(0, len1):
	total_latency.append((float)(trace1[i]))

total_latency.sort()

sample_interval = (int) (len(total_latency) / sample_rate)
sample_array = []

for i in range(0, len1, sample_interval):
	sample_array.append(total_latency[i])
sample_array.append(total_latency[len1-1])

for i in range(0, len(sample_array)):
	print sample_array[i]
