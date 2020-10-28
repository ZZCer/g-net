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
#k1=36.167/128
#b1=102
#k2=10.905/128
#b2=16.229
#minB=128
k1=0.570586
b1=176.86
k2=0.172031
b2=11.496

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

B0 = 186
N=20
L = k1 * B0 + b1 + k2 * N * B0 + b2
batch = B0 * N
T = batch / L
print str(N) + "\tT:" + str(T) + "\tL:" + str(L)
print str(k1*B0+b1)
