#!/usr/bin/python
import os
import math
import sys

# k1 and b1 are for kernel execution
# k2 and b2 are for memory transfer
#NIDS - 512B packet
#k1=13.13/128
#b1=272.7
#k2=10.068/128
#b2=19.632
#minB=128

#IPSEC - 512B packet
k1=36.167/128
b1=102
k2=10.905/128
b2=16.229
minB=128

#ROUTER
#k1=0.00109625
#b1=8.425
#k2=0.0004133
#b2=2.8036
#minB=512

#firewall
#k1=0.136325
#b1=15.241
#k2=0.005968
#b2=9.3574
#minB=512

L=1000
#T=14.88 * 1
T=2.35 * 2

P_PERF=0.5
T = (1 + P_PERF) * T

for N in range(1, 29, 1):
	# T = B0*N/L = B0*N/(k1*B0+b1+k2*N*B0+b2)
	B = math.ceil(T*(b1+b2) / (N-T*k1-T*k2*N))
	if B < 0:
		continue
	if B < minB:
		B = minB
	L = k1*B + b1 + k2*N*B + b2

	#print "T0 = " + str(T) + ",\tB = " + str(B) + ",\tL = " + str(L)
	print "[" + str(N) + "]\t" + str(L) + "\t" + str(B)
		#print str(N) + '\t' + str(L)
