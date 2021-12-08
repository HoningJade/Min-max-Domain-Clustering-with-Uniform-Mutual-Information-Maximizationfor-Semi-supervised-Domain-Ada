#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py \
  --method MME --dataset multi \
  --source real --target sketch \
  --net resnet34 --save_check --checkpath /datab/duyxxd/DNet/uni_jsd_truesrc --bs 24 \
  --momentum 0.99 --use_true_src --smlbd $2
