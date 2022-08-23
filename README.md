# Understanding Collapse in Non-Contrastive Siamese Representation Learning

[Alexander C. Li](http://alexanderli.com/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/). In ECCV 2022.

This is a repository to analyze partial dimensional collapse in non-contrastive pre-training. This codebase is inspired by the [SimSiam](https://github.com/facebookresearch/simsiam) and [MoCo-v3]() repos.   
## Setup 
```bash
conda env create -f environment.yml
```
You will additionally need to [install the correct version of PyTorch](https://pytorch.org/get-started/locally/)
depending on your system. Finally, download and set up ImageNet based on [these instructions](https://github.com/pytorch/examples/tree/main/imagenet).   
## Training
### SimSiam Pre-training
```bash
# IID
python main_simsiam.py -a resnet18 --fix-pred-lr --epochs 100 --workers 10 -b 256 --dataset_type imagenet_train /home/datasets/imagenet \
  --gpu 0 --seed 10 --exp_name rn18-imagenet-multiepoch --exp_id 1

# Single pass
python main_simsiam.py -a resnet18 --fix-pred-lr --epochs 100 --workers 10 -b 256 --dataset_type imagenet_train /home/datasets/imagenet \
  --gpu 0 --seed 10 --incremental_idx_path autolearn/simsiam/incremental_imagenet_ordering.npy \
  --exp_name rn18-imagenet-singlepass --exp_id 2 --strategy current 

# Single pass, cumulative 
python main_simsiam.py -a resnet18 --fix-pred-lr --epochs 100 --workers 10 -b 256 --dataset_type imagenet_train /home/datasets/imagenet \
  --gpu 0 --seed 10 --incremental_idx_path autolearn/simsiam/incremental_imagenet_ordering.npy \
  --exp_name rn18-imagenet-cumulative --exp_id 3 --strategy accumulate 

# Train on percentage (e.g., 20%) of dataset
python main_simsiam.py -a resnet18 --fix-pred-lr --epochs 100 --workers 10 -b 256 --dataset_type imagenet_train /home/datasets/imagenet \
  --gpu 0 --seed 10 --incremental_idx_path autolearn/simsiam/incremental_imagenet_ordering.npy \
  --exp_name rn18-imagenet-last20 --exp_id 4 --strategy last --chunk_size 20

# Hybrid (multi-epoch training using single pass checkpoint)
python main_simsiam.py -a resnet18 --fix-pred-lr --epochs 100 --workers 10 -b 256 --dataset_type imagenet_train /home/datasets/imagenet \
  --gpu 0 --seed 10 --exp_name rn18-imagenet-multiepoch --exp_id 5 \
  --resume logs/rn18-imagenet-singlepass/files/checkpoint_0039.pth.tar  
```

### Linear Probe
To run a linear probe on ImageNet using a pre-trained ResNet-18 checkpoint: 
```bash
python main_lincls.py -a resnet18 --gpu 0 --lars --dataset_type imagenet --workers 9 -b 1024 /home/datasets/imagenet \
  --exp_name resnet18_lp --exp_id 1 --pretrained logs/resnet18_simsiam/files/checkpoint_0099.pth.tar  
```
## Computing Metrics
### k-NN Accuracy 
```bash
python save_reprs_clean.py -a resnet18 --batch-size 1024 --gpu 0 --dataset_type imagenet_train --exp_id 1 --data /home/datasets/imagenet \
     --pretrained logs/resnet18_simsiam/files/checkpoint_0099.pth.tar  
python save_reprs_clean.py -a resnet18 --batch-size 1024 --gpu 0 --dataset_type imagenet_val --exp_id 1 --data /home/datasets/imagenet \
     --pretrained logs/resnet18_simsiam/files/checkpoint_0099.pth.tar  
python compute_nn.py --folder data/resnet18_exp1/imagenet --source train_reprs_e0.pt --target val_reprs_e0.pt --metric cos
```
Alternatively, we have provided a bash script to automatically compute the k-NN accuracy,
provided the architecture type and the experiment log directory. 
```bash 
bash knn_eval.sh resnet18 logs/resnet18_simsiam
```
### Collapse 
```bash 
python save_reprs_clean.py -a resnet18 --batch-size 1024 --gpu 0 --dataset_type imagenet_train --exp_id 1 --data /home/datasets/imagenet \
     --pretrained logs/resnet18_simsiam/files/checkpoint_0099.pth.tar  
python test_collapse.py data/resnet18_exp1/imagenet/train_reprs_e0.pt --type svd
```
A notebook is provided to load, plot, and compute the resulting collapse metrics. 
## Citation 
```
@article{SimSiamCollapse,
    title={Understanding Collapse in Non-Contrastive Siamese Representation Learning},
    author={Li, Alexander Cong and Efros, Alexei A. and Pathak, Deepak},
    journal={ECCV},
    year={2022}
}
``` 