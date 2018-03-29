#!/bin/bash

ncpu=$(grep -c '^processor' /proc/cpuinfo)

pushd "$(dirname $(readlink -f $0))"
source env.rc
cd "$RTE_SDK"
make -j$ncpu config T=$RTE_TARGET
make -j$ncpu T=$RTE_TARGET
make -j$ncpu install T=$RTE_TARGET
popd
