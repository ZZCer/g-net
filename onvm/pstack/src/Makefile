#
# Makefile for libnids.
#
# Kai Zhang <kay21s@gmail.com>
# Dug Song <dugsong@monkey.org>

srcdir		= .

LIBSTATIC      = libnids.a

CC		= gcc
# Note: Major location can only be used when COMPACT_TABLE is enabled or INDEXFREE is enabled
# CRC_SIGN can only be used when msse4.2 is enabled
CFLAGS		= -O2 -lpthread -D_BSD_SOURCE -DLIBNET_VER=-1 -DHAVE_ICMPHDR=1 -DHAVE_TCP_STATES=1 -DHAVE_BSD_UDPHDR=1 
CFLAGS		+= -D__USE_GNU -D_BSD_SOURCE
CFLAGS		+= -DORIGIN_TCP # -DMAJOR_INDEXFREE_TCP  # Only one should be enabled
CFLAGS		+= -DBUILTIN_FUNC -DPRE_MEM -DDISABLE_UPPER_LAYER
CFLAGS		+= -DCRC_HASH_0 -DMULTIPLICATION_HASH_0 -DCRC_SIGN2 -msse4.2 
CFLAGS		+= -DPARALLEL
CFLAGS		+= -DUSE_ULCC_0 -DMAJOR_LOCATION_0 #-DCRC_MAJOR #-DHASH_MAJOR
CFLAGS		+= -DMEM_LL_0 -DMEM_ALIGN
CFLAGS		+= -DDEBUG_0 -DCYCLE_0


#PCAP_CFLAGS	= -I/usr/local/include
#PCAPLIB		= -L/usr/local/lib -lpcap

#LIBS		= -lnsl -pthread 

INC += -I../../ -I../../../onvm_nflib -I.

# Sequential version, Open this
#OBJS		= checksum.o ip_fragment.o ip_options.o killtcp.o \
		  libnids.o scan.o tcp.o util.o allpromisc.o hash.o \
		  bitmap.o conn_attribute.o \
		  conn_major_indexfree.o
# Parallel version, Open this
OBJS		= checksum.o ip_fragment.threaded.o ip_options.o killtcp.o \
		  libnids.o scan.o tcp.threaded.o util.o allpromisc.o hash.o \
		  bitmap.threaded.o conn_attribute.o fifo.o mem.o \
		  conn_origin.threaded.o # conn_major_indexfree.threaded.o

.c.o:
	$(CC) -c $(CFLAGS) $(INC) $<
static: $(LIBSTATIC)

$(LIBSTATIC): $(OBJS)
	ar -cr $@ $(OBJS)

clean:
	rm -f *.o *~ $(LIBSTATIC) $(LIBSHARED)

# EOF
