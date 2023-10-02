#!/bin/zsh
source /opt/intel/oneapi/setvars.sh 
export OMP_NUM_THREADS=64
make clean && make -j 2
export CUDA_VISIBLE_DEVICES=1
./main.exe