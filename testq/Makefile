ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

RTE_TARGET ?= x86_64-native-linuxapp-gcc
CUDA_PATH  ?= /usr/local/cuda

# Default target, can be overriden by command line or environment
include $(RTE_SDK)/mk/rte.vars.mk

# binary name
APP = testq

# all source are stored in SRCS-y
SRCS-y := testq.c

# workaround for a gcc bug with noreturn attribute
# http://gcc.gnu.org/bugzilla/show_bug.cgi?id=12603
ifeq ($(CONFIG_RTE_TOOLCHAIN_GCC),y)
CFLAGS_testq.o += -Wno-return-type -O3
endif

include $(RTE_SDK)/mk/rte.extapp.mk
