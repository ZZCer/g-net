#!/bin/bash

cpu_num=$(expr $1 - 1)
#cpu="12"
#cpu_end=$(expr $cpu_num + 12)
#for k in $( seq 13 $cpu_end)
cpu="0"
for k in $( seq 1 $cpu_num)
do
        cpu="$cpu","$k"
done
ports=1

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH

if [ -z $ports ]
then
        echo "$0 [cpu-list] [port-bitmask]"
        # this works well on our 2x6-core nodes
        echo "$0 0,1,2,6 3 --> cores 0, 1, 2 and 6 with ports 0 and 1"
        echo "Cores will be used as follows in numerical order:"
        echo "  RX thread, TX thread, ..., TX thread for last NF, Stats thread"
        exit 1
fi

sudo rm -rf /mnt/huge/*
sudo rm -rf /dev/hugepages/*
echo $SCRIPTPATH/onvm_mgr/$RTE_TARGET/onvm_mgr -l $cpu -n 4 --proc-type=primary --log-level 7 -- -p${ports}
#sudo $SCRIPTPATH/onvm_mgr/$RTE_TARGET/onvm_mgr -l $cpu -n 4 --proc-type=primary --log-level 7 -- -p${ports}
sudo $SCRIPTPATH/onvm_mgr/$RTE_TARGET/onvm_mgr -l $cpu -n 4 --proc-type=primary --log-level 7 -- -p${ports}
