#!/bin/bash
sudo modprobe uio
sudo insmod dpdk/build/kmod/igb_uio.ko
sudo ip link set enp2s0f0 down
sudo dpdk/usertools/dpdk-devbind.py -b igb_uio enp2s0f0

