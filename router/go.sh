#!/bin/bash

function usage {
        echo "$0 CPU_START WORKER_THREAD_NUM"
        echo "$0 3 4 --> cores 3,4,5,6 are used for CPU worker, core 7 is used for Scheduler"
        exit 1
}

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cpu=$1
thread_num=$2

cpu_list=$cpu
for k in $(seq 1 $thread_num)
do
    new=$(expr $cpu + $((k*2)))
        cpu_list="$cpu_list","$new"
done

shift 3

#exec sudo $SCRIPTPATH/build/router -l $cpu_list -n 4 --proc-type=secondary --base-virtaddr=0x7fffdc200000 --log-level 7 -- -k $thread_num
exec sudo $SCRIPTPATH/build/router -l 4,6 -n 4 --proc-type=secondary --log-level 7 -- -k 1
