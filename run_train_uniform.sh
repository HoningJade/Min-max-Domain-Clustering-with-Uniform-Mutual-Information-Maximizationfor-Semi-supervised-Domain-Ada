#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py \
  --method MME --dataset multi \
  --source real --target sketch \
  --net resnet34 --save_check --checkpath /datab/duyxxd/DNet/uniform --bs 24 --uniform_sampling --pseudo_balance_target
