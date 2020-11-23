#                    openNetVM
#      https://github.com/sdnfv/openNetVM
#
# BSD LICENSE
#
# Copyright(c)
#          2015-2016 George Washington University
#          2015-2016 University of California Riverside
#          2010-2014 Intel Corporation.
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

# subsystem:
#	cd pstack && $(MAKE) clean && $(MAKE)

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

# Default target, can be overriden by command line or environment
RTE_TARGET ?= x86_64-native-linuxapp-gcc
CUDA_PATH  ?= /usr/local/cuda
CJSON_PATH ?= /usr/local/cJSON

include $(RTE_SDK)/mk/rte.vars.mk

ifneq ($(CONFIG_RTE_EXEC_ENV),"linuxapp")
$(error This application can only operate in a linuxapp environment, \
please change the definition of the RTE_TARGET environment variable)
endif

# binary name
APP = onvm_mgr

# all source are stored in SRCS-y
SRCS-y := main.c onvm_init.c onvm_args.c onvm_stats.c onvm_nf.c manager.c scheduler.c pstack.c record.c synchronize.c 

INC :=

WERROR_FLAGS += -Wno-error=unused-function -Wno-error=maybe-uninitialized


#cJSON lib
#CFLAGS中通过-I指令指定头文件目录
#LDFLAGS -L指定库文件路径 -l指定所使用的库
LDFLAGS += -L$(CJSON_PATH)/build/lib  -lcjson 

# CFLAGS += $(WERROR_FLAGS) -O3 -g -rdynamic $(USER_FLAGS)
CFLAGS += $(WERROR_FLAGS) -O3 -g -rdynamic $(USER_FLAGS)

CFLAGS += -I$(SRCDIR)/../onvm_nflib
LDFLAGS += -L$(SRCDIR)/../onvm_nflib/$(RTE_TARGET)/lib -lonvm

CFLAGS += -I$(CUDA_PATH)/include
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcuda

CFLAGS += -I$(SRCDIR)/../pstack/src -DPARALLEL -DORIGIN_TCP
LDFLAGS += -L$(SRCDIR)/../pstack/src -lnids

# for newer gcc, e.g. 4.4, no-strict-aliasing may not be necessary
# and so the next line can be removed in those cases.
EXTRA_CFLAGS += -fno-strict-aliasing

include $(RTE_SDK)/mk/rte.extapp.mk
