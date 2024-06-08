#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt.1_val5_8w_randInit_Softmax

CUDA_VISIBLE_DEVICES=$1 python train.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=2 \
--num_gpus=1 \
--num_steps=80000 \
--val_pred_every=100 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS18/BraTS18_metaTrain.csv \
--val_list=BraTS18/BraTS18_metaVal.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--random_seed=999 \
--reload_path=snapshots/final.pth \
--reload_from_checkpoint=False > logs/${time}_train_${name}.log


time=$(date "+%Y%m%d-%H%M%S")
name=Missing_11.5w_BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt.1_val5_8w_randInit_Softmax

CUDA_VISIBLE_DEVICES=$1 python train.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=2 \
--num_gpus=1 \
--num_steps=115000 \
--val_pred_every=500 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS18/BraTS18_metaTrain.csv \
--val_list=BraTS18/BraTS18_metaVal.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--random_seed=999 \
--reload_path=snapshots/BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt.1_val5_8w_randInit_Softmax/final.pth \
--reload_from_checkpoint=True \
--mode=random > logs/${time}_train_${name}.log