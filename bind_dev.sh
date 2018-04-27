#!/bin/bash
sudo modprobe uio
sudo insmod dpdk/build/kmod/igb_uio.ko
sudo ip link set p1p2 down
sudo dpdk/usertools/dpdk-devbind.py -b igb_uio p1p2

