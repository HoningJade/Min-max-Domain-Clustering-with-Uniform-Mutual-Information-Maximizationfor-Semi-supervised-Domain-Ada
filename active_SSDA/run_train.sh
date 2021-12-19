#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --dataset office_home --source Real --target Clipart --net $3 --save_check
