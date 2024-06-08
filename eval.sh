#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=Eval_BraTS18_SS_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_L1

CUDA_VISIBLE_DEVICES=$1 python eval.py \
--input_size=80,160,160 \
--num_classes=3 \
--data_list=BraTS18/BraTS18_test.csv \
--weight_std=True \
--restore_from=snapshots/Eval_BraTS18_SS_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_L1/final.pth \
--mode=1,2,3,4 > logs/${time}_train_${name}.log 2>&1 &
