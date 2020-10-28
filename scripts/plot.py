#!/usr/bin/python
import plotly.plotly as py
import plotly as py1
from plotly.graph_objs import *
import sys

# plot the traces from both the Manager and NFs, for comparison
# Manager shows the exact time of each memcpy and kernrel execution
# Each NF trace shows the overall time of processing a batch

# the trace file from the Manager is named "t"
# the trace file from NF i is named "ti"
trace_file = "t"
trace_num = int(1)

traces = open(trace_file, "r").readlines()

trace_x = [[],[],[],[],[]]
trace_y = [[],[],[],[],[]]

htod_height = 1
kernel_height = 3
dtoh_height = 2
all_height = 4

base_height_para = 5

for line in traces:
	line = line.split()
	if len(line) == 0 or line[0] < '0' or line[0] > '9':
		continue
	base_height = int(line[0]) * base_height_para # instance_id * 5 is the height
	op_type = int(line[1])
	#print base_height, op_type, line
	if op_type == 1: # htod start
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height)
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height + htod_height)
	elif op_type == 2: # htod end
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height + htod_height)
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height)
	elif op_type == 3: # kernel start
		trace_x[int(line[0])].append(float(line[2]))
		trace_y[int(line[0])].append(base_height)
		trace_x[int(line[0])].append(float(line[2]))
		trace_y[int(line[0])].append(base_height + kernel_height)
	elif op_type == 4: # kernel end
		trace_x[int(line[0])].append(float(line[2]))
		trace_y[int(line[0])].append(base_height + kernel_height)
		trace_x[int(line[0])].append(float(line[2]))
		trace_y[int(line[0])].append(base_height)
	elif op_type == 5: # dtoh start
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height)
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height + dtoh_height)
	elif op_type == 6: # dtoh end
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height + dtoh_height)
		trace_x[int(line[0])].append(float(line[3]))
		trace_y[int(line[0])].append(base_height)
	else:
		print "error: " + line

for i in range(1, trace_num + 1, 1):
	trace_file = "t"+str(i)
	traces = open(trace_file, "r").readlines()
	for line in traces:
		line = line.split()
		if len(line) == 0 or line[0] < '0' or line[0] > '9':
			continue
		base_height = int(line[0]) * base_height_para
		op_type = int(line[1])
		if op_type == 7:
			trace_x[int(line[0])].append(float(line[2]))
			trace_y[int(line[0])].append(base_height)
			trace_x[int(line[0])].append(float(line[2]))
			trace_y[int(line[0])].append(base_height + all_height)
		elif op_type == 8:
			trace_x[int(line[0])].append(float(line[2]))
			trace_y[int(line[0])].append(base_height + all_height)
			trace_x[int(line[0])].append(float(line[2]))
			trace_y[int(line[0])].append(base_height)
		else:
			print "error: " + line


my_data = [[],[],[],[]]
for i in range(1,5,1):
	my_data[i-1] = Scatter(
		x = trace_x[i],
		y = trace_y[i]
	)


data = Data(my_data)

py1.offline.plot(data, filename='time_stat.html')
