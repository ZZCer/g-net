#!/usr/bin/python
import os
import math
import sys

# k1 and b1 are for kernel execution
# k2 and b2 are for memory transfer

#NIDS - 512B packet
k1=13.13/128
b1=272.7
k2=10.068/128
b2=19.632
minB=128

#IPSEC - 512B packet
#k1=36.167/128
#b1=102
#k2=10.905/128
#b2=16.229
#minB=128

#Router
#k1=0.00109625
#b1=8.425
#k2=0.0004133
#b2=2.8036
#minB=512

#Firewall
#k1=0.136325
#b1=15.241
#k2=0.005968
#b2=9.3574
#minB=512

# For Router and Firewall
#L=200
#T=14.88*8

# For NIDS and IPSec
L=1000
T=2.35 * 3

P_PERF=0.0
T = (1 + P_PERF) * T

for N in range(1, 29, 1):
	# T = B0*N/L = B0*N/(k1*B0+b1+k2*N*B0+b2)
	B = math.ceil(T*(b1+b2) / (N-T*k1-T*k2*N))
	if B < 0:
		continue;
	if B < minB:
		B = minB
	L0 = k1*B + b1 + k2*N*B + b2
	if L0 < L:
		break

print "Num of SMs: " + str(N) + ", Batch size is " + str(B) + ", L is " + str(L0) + "\n"

sm_num = N + 1

preL = L0

for N in range(sm_num, 29, 1):
	B = math.ceil(T*(b1+b2) / (N-T*k1-T*k2*N))
	if B < minB:
		B = minB
	L0 = k1*B + b1 + k2*N*B + b2
	if preL - L0 < 5:
	 	print "Limited latency improvement, break. B = " +  str(B) + ", L0 = " + str(L0)
		break

	print "[SM " + str(N) + "]\tpreL is " + str(preL) + ", new L is " + str(L0) + ", B is " + str(B)
	preL = L0
