#!/bin/bash

function usage {
        echo "$0 CPU-LIST SERVICE-ID DST [-p PRINT] [-n NF-ID]"
        echo "$0 3,7,9 1 2 --> cores 3,7, and 9, with Service ID 1, and forwards to service ID 2"
        echo "$0 3,7,9 1 2 1000 --> cores 3,7, and 9, with Service ID 1, forwards to service ID 2,  and Print Rate of 1000"
        exit 1
}

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cpu=$1
thread_num=$2

shift 3

exec sudo $SCRIPTPATH/build/raw_framework -l $cpu -n 4 --proc-type=secondary --log-level 7 -- -k $thread_num
