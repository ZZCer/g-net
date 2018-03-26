ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

RTE_TARGET ?= x86_64-native-linuxapp-gcc

# Default target, can be overriden by command line or environment
include $(RTE_SDK)/mk/rte.vars.mk

# binary name
APP = ipsec

# all source are stored in SRCS-y
SRCS-y := ipsec.c

# OpenNetVM path
ONVM= $(SRCDIR)/../onvm

CFLAGS += $(WERROR_FLAGS) -g $(USER_FLAGS)

CFLAGS += -I./gpu
CFLAGS += -I$(ONVM)/onvm_nflib
CFLAGS += -I/usr/local/cuda/include
LDFLAGS += $(ONVM)/onvm_nflib/onvm_nflib/$(RTE_TARGET)/onvm_nflib.o
LDFLAGS += $(ONVM)/onvm_nflib/onvm_nflib/$(RTE_TARGET)/onvm_pkt_helper.o
LDFLAGS += $(ONVM)/onvm_nflib/onvm_nflib/$(RTE_TARGET)/onvm_sc_common.o
LDFLAGS += $(ONVM)/onvm_nflib/onvm_nflib/$(RTE_TARGET)/onvm_framework.o
LDFLAGS += $(ONVM)/shared/shared/$(RTE_TARGET)/fifo.o
LDFLAGS += -L/usr/local/cuda/lib64 -lcudart -lcuda

# workaround for a gcc bug with noreturn attribute
# http://gcc.gnu.org/bugzilla/show_bug.cgi?id=12603
ifeq ($(CONFIG_RTE_TOOLCHAIN_GCC),y)
CFLAGS_main.o += -Wno-return-type
endif

include $(RTE_SDK)/mk/rte.extapp.mk
