#!/usr/bin/env bash
# bash knn_eval.sh ARCH FOLDER_PATH
exp_id=${2: -4}
python save_reprs_clean.py -a $1 --batch-size 1024 --gpu 0 --dataset_type imagenet_train --exp_id $exp_id --data /home/alexli/datasets/imagenet \
     --pretrained $2/files/checkpoint_0099.pth.tar
python save_reprs_clean.py -a $1 --batch-size 1024 --gpu 0 --dataset_type imagenet_val --exp_id $exp_id --data /home/alexli/datasets/imagenet \
     --pretrained $2/files/checkpoint_0099.pth.tar
#python compute_nn.py --folder data/$1_exp$exp_id/imagenet --source train_reprs_e0.pt --target val_reprs_e0.pt --metric l2
python compute_nn.py --folder data/$1_exp$exp_id/imagenet --source train_reprs_e0.pt --target val_reprs_e0.pt --metric cos
