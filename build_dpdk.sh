#!/bin/bash

pushd "$(dirname $0)"
source env.sh
cd dpdk
make -j4 config T=$RTE_TARGET
make -j4 T=$RTE_TARGET
popd
