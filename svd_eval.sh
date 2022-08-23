#!/usr/bin/env bash
# bash knn_eval.sh ARCH PATH
exp_id=${2: -4}
python save_reprs_clean.py -a $1 --batch-size 1024 --gpu 0 --dataset_type imagenet_train --exp_id $exp_id --data /home/alexli/datasets/imagenet \
     --pretrained $2/files/checkpoint_0099.pth.tar
python test_collapse.py data/$1_exp$exp_id/imagenet/train_reprs_e0.pt --type svd
