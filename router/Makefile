ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

RTE_TARGET ?= x86_64-native-linuxapp-gcc
CUDA_PATH  ?= /usr/local/cuda

# Default target, can be overriden by command line or environment
include $(RTE_SDK)/mk/rte.vars.mk

# binary name
APP = router

# all source are stored in SRCS-y
SRCS-y := router.c

# OpenNetVM path
ONVM= $(SRCDIR)/../onvm

CFLAGS += $(WERROR_FLAGS) -g $(USER_FLAGS)

CFLAGS += -I$(ONVM)/onvm_nflib
CFLAGS += -I$(CUDA_PATH)/include
LDFLAGS += -L$(ONVM)/onvm_nflib/$(RTE_TARGET) -lonvm
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcuda

# workaround for a gcc bug with noreturn attribute
# http://gcc.gnu.org/bugzilla/show_bug.cgi?id=12603
ifeq ($(CONFIG_RTE_TOOLCHAIN_GCC),y)
CFLAGS_main.o += -Wno-return-type
endif

include $(RTE_SDK)/mk/rte.extapp.mk
