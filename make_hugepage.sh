#!/bin/bash

echo 128 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
echo 16 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
