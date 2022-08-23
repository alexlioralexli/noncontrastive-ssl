#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
from copy import deepcopy
import gc
import os
import random
import warnings
import wandb
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import autolearn.simsiam.builder as builder
from autolearn.simsiam.data import imagenet_train, imagenet_train_subset, places_simsiam_train, places_train_subset
import resnet_variants

from main_simsiam import (zero_encoder_lr, AverageMeter, ProgressMeter, reset_encoder_lr, model_names,
                          adjust_learning_rate, save_checkpoint, get_incremental_dataloader)
from save_reprs_clean import create_and_load_model

models.__dict__.update(resnet_variants.__dict__)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--teacher_arch', metavar='ARCH', default='resnet50', type=str)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--final_lr', default=0.0, type=float,
                    metavar='LR', help='final (base) learning rate', dest='final_lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--teacher_checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=10, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

# added
parser.add_argument('--dataset_type', default='imagenet_train', type=str)
parser.add_argument('--n_augs', default=2, type=int)
parser.add_argument('--idx_path', default=None, type=str)
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--small', action='store_true', help='Use small version of Places')
parser.add_argument('--test', '-t', action='store_true')
parser.add_argument('--exp_id', default=-1, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--pretrained_distributed', action='store_true',
                    help='was the pretrained checkpoint distributed?')

# for incremental
parser.add_argument('--incremental_idx_path', default=None, type=str)
parser.add_argument('--strategy', default='iid', type=str)
parser.add_argument('--chunk_size', default=1, type=int)
parser.add_argument('--repeat_batch', default=1, type=int)
parser.add_argument('--n_warmup_steps', default=0, type=int)
parser.add_argument('--steps_per_epoch', default=None, type=int)
parser.add_argument('--n_rounds', default=1, type=int)
parser.add_argument('--fix_all_lr', action='store_true')


# for distill
parser.add_argument('--teacher_repr_path', default=None, type=str, help="Path to folder with teacher representations")
parser.add_argument('--ema_alpha', default=0.05, type=float)
parser.add_argument('--alg', type=str, default='simsiam')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu == 0:
        exp_name = args.exp_name if args.exp_name is not None else args.dataset_type
        if args.test:
            os.environ['WANDB_MODE'] = 'offline'
        wandb.init(config=vars(args), id=f'{exp_name}_{args.exp_id:04d}', project='simsiam', resume=not args.no_resume,
                   settings=wandb.Settings(start_method="fork"))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating teacher model '{}'".format(args.teacher_arch))
    assert os.path.isfile(args.teacher_checkpoint)
    teacher_args = deepcopy(args)
    teacher_args.pretrained = teacher_args.teacher_checkpoint
    teacher_args.arch = teacher_args.teacher_arch
    teacher = create_and_load_model(teacher_args)
    teacher_mean, teacher_std = None, None

    print("=> creating student model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=2048, zero_init_residual=True)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    final_lr = args.final_lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            teacher.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            teacher.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        teacher = teacher.cuda(args.gpu)
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model)  # print model after SyncBatchNorm

    if args.fix_pred_lr:
        if args.distributed:
            raise NotImplementedError
        else:
            optim_params = [{'params': model.parameters(), 'fix_lr': False, 'is_encoder': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    checkpoint_file = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint_file = args.resume
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif (not args.distributed) and wandb.run.resumed:
        print("=> Trying to resume from wandb")
        try:
            wandb.restore('checkpoint.pth.tar')
            checkpoint_file = os.path.join(wandb.run.dir, 'checkpoint.pth.tar')
        except ValueError:
            pass
    if checkpoint_file:
        if args.gpu is None:
            checkpoint = torch.load(checkpoint_file)
        else:
            # Map model to be loaded to specified single gpu.
            # loc = 'cuda:{}'.format(args.gpu)
            # checkpoint = torch.load(args.resume, map_location=loc)
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        if args.start_epoch == 0:
            args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        teacher_mean = checkpoint['teacher_mean'].cuda(args.gpu)
        teacher_std = checkpoint['teacher_std'].cuda(args.gpu)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    cudnn.benchmark = True

    # Data loading code
    if args.n_augs != 2:
        assert args.dataset_type == 'imagenet_train'
    if args.dataset_type == 'imagenet_train':
        train_dataset = imagenet_train(args, n_augs=args.n_augs)
    elif args.dataset_type == 'imagenet_train_subset':
        train_dataset = imagenet_train_subset(args)
    elif args.dataset_type == 'places_simsiam_train':
        train_dataset = places_simsiam_train(args)
    elif args.dataset_type == 'places_train_subset':
        train_dataset = places_train_subset(args)
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if teacher_mean is None and teacher_std is None:
        teacher_mean, teacher_std = compute_normalization_stats(args)
    for epoch in range(args.start_epoch, args.epochs):
        if args.strategy != 'iid':
            if args.multiprocessing_distributed:
                total_batch_size = args.batch_size * ngpus_per_node
            else:
                total_batch_size = args.batch_size
            train_loader, train_sampler = get_incremental_dataloader(gpu,
                                                                     train_dataset,
                                                                     epoch,
                                                                     total_batch_size,
                                                                     args)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, final_lr, epoch, args)

        # train for one epoch
        train_stats, teacher_mean, teacher_std = train(train_loader, teacher, model, teacher_mean,
                                                       teacher_std, optimizer, epoch, args)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            wandb.log(train_stats)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'teacher_mean': teacher_mean,
                'teacher_std': teacher_std
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch), folder=wandb.run.dir)
            wandb.save(os.path.join(wandb.run.dir, 'checkpoint.pth.tar'))

    if args.multiprocessing_distributed and args.rank == 0:
        wandb.finish()


def train(train_loader, teacher, student, teacher_mean, teacher_std, optimizer, epoch, args):
    print('Training, epoch', epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    teacher.eval()
    student.train()

    if args.n_warmup_steps > 0 and epoch != 0:
        print('Turning grad off for encoder')
        zero_encoder_lr(optimizer)

    end = time.time()
    for i, (imgs, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.n_warmup_steps > 0 and args.n_warmup_steps == i and epoch != 0:
            print('Turning grad back on for encoder')
            reset_encoder_lr(optimizer)

        img = imgs[0]
        if args.gpu is not None:
            img = img.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        with torch.no_grad():
            teacher_pred = teacher(img)
            new_teacher_pred = (teacher_pred - teacher_mean) / (1e-6 + teacher_std)
            teacher_mean = args.ema_alpha * teacher_pred.mean(dim=0) + (1 - args.ema_alpha) * teacher_mean
            teacher_std = args.ema_alpha * teacher_pred.std(dim=0) + (1 - args.ema_alpha) * teacher_std
        student_pred = student(img)
        loss = F.mse_loss(student_pred, new_teacher_pred)
        losses.update(loss.item(), len(img))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return dict(batch_time=batch_time.avg,
                data_time=data_time.avg,
                loss=losses.avg), teacher_mean, teacher_std

def compute_normalization_stats(args):
    if args.teacher_repr_path:
        print(f"Computing teacher representation mean/std from {args.teacher_repr_path}")
        representations = torch.load(args.teacher_repr_path)['representations']
        mean = representations.mean(dim=0, keepdims=True).cuda(args.gpu)
        std = representations.std(dim=0, keepdims=True).cuda(args.gpu)
        print(f"Average mean: {mean.mean()}, average std: {std.mean()}")
    else:
        print('Initializing teacher mean=0 and std=1')
        mean = torch.zeros(1, 2048).cuda(args.gpu)
        std = torch.ones(1, 2048).cuda(args.gpu)
    return mean, std



if __name__ == '__main__':
    main()
