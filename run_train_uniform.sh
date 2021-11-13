#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py \
  --method $2 --dataset multi \
  --source real --target sketch \
  --net $3 --save_check --checkpath $4 --bs 24 --uniform_sampling --pseudo_balance_target
