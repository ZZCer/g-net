#                    openNetVM
#      https://github.com/sdnfv/openNetVM
#
# BSD LICENSE
#
# Copyright(c)
#          2015-2016 George Washington University
#          2015-2016 University of California Riverside
#          2010-2014 Intel Corporation.
#          2016 Hewlett Packard Enterprise Development LP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
# The name of the author may not be used to endorse or promote
# products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

# Default target, can be overriden by command line or environment
include $(RTE_SDK)/mk/rte.vars.mk

RTE_TARGET ?= x86_64-native-linuxapp-gcc
CUDA_PATH  ?= /usr/local/cuda

# binary name
LIB    = libonvm.a

# all source are stored in SRCS-y
SRCS-y := onvm_pkt_helper.c onvm_nflib.c onvm_framework.c onvm_sc_mgr.c onvm_sc_common.c \
			 ../onvm_mgr/pstack.c

INC := onvm_includes.h

CFLAGS += -I$(SRCDIR)/../pstack/src -DPARALLEL -DORIGIN_TCP -lnids

# CFLAGS += $(WERROR_FLAGS) -O3 -g -rdynamic $(USER_FLAGS)
CFLAGS += -O3 -g -rdynamic $(USER_FLAGS)
CFLAGS += -I$(SRCDIR)
CFLAGS += -I$(CUDA_PATH)/include
CFLAGS += -D__CUDA_ARCH__=0

include $(RTE_SDK)/mk/rte.extlib.mk
