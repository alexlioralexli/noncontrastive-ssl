#!/usr/bin/env bash
# bash knn_eval.sh ARCH CHECKPOINT PATH
exp_id=${3: -4}
python save_reprs_clean.py -a $1 --batch-size 1024 --gpu 0 --dataset_type imagenet_train --exp_id $exp_id --data /home/alexli/datasets/imagenet \
     --pretrained $3/files/checkpoint_$2.pth.tar --extra_info ckpt$2
python save_reprs_clean.py -a $1 --batch-size 1024 --gpu 0 --dataset_type imagenet_val --exp_id $exp_id --data /home/alexli/datasets/imagenet \
     --pretrained $3/files/checkpoint_$2.pth.tar  --extra_info ckpt$2
#python compute_nn.py --folder data/$1_exp${exp_id}_ckpt$2/imagenet --source train_reprs_e0.pt --target val_reprs_e0.pt --metric l2
python compute_nn.py --folder data/$1_exp${exp_id}_ckpt$2/imagenet --source train_reprs_e0.pt --target val_reprs_e0.pt --metric cos
