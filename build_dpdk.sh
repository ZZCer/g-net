#!/bin/bash

ncpu=$(grep -c '^processor' /proc/cpuinfo)

pushd "$(dirname $0)"
source env.rc
cd dpdk
make -j$ncpu config T=$RTE_TARGET
make -j$ncpu T=$RTE_TARGET
make -j$ncpu install T=$RTE_TARGET
popd
