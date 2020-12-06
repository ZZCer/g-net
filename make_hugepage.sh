#!/bin/bash

echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
echo 32 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
