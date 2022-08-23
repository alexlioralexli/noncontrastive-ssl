#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import gc
import os
import random
import warnings
import wandb
import math
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import autolearn.simsiam.builder as builder
from torch.utils.data import Subset
from autolearn.simsiam.data import imagenet_train, imagenet_train_subset, places_simsiam_train, places_train_subset
from autolearn.simsiam.fast_simsiam_error import fast_simsiam_error
import resnet_variants
from autolearn import vits
from functools import partial
# Slurm requeueing
import signal

# MAIN_PID = os.getpid()
# SIGNAL_RECEIVED = False
#
#
# def SIGTERMHandler(a, b):
#     print('received sigterm')
#     pass
#
#
# def signalHandler(a, b):
#     global SIGNAL_RECEIVED
#     print('Signal received', a, time.time(), flush=True)
#     SIGNAL_RECEIVED = True
#     trigger_job_requeue()
#     return
#
#
# def trigger_job_requeue():
#     ''' Submit a new job to resume from checkpoint.
#     '''
#     if os.environ['SLURM_PROCID'] == '0' and \
#             os.getpid() == MAIN_PID:
#         ''' BE AWARE OF subprocesses that your program spawns.
#         Only the main process on slurm procID = 0 resubmits the job.
#         In pytorch imagenet example, by default it spawns 4
#         (specified in -j) subprocesses for data loading process,
#         both parent process and child processes will receive the signal.
#         Please only submit the job in main process,
#         otherwise the job queue will be filled up exponentially.
#         Command below can be used to check the pid of running processes.
#         print('pid: ', os.getpid(), ' ppid: ', os.getppid(), flush=True)
#         '''
#         print('time is up, back to slurm queue', flush=True)
#         command = 'scontrol requeue ' + os.environ['SLURM_JOB_ID']
#         print(command)
#         if os.system(command):
#             raise RuntimeError('requeue failed')
#         print('New job submitted to the queue', flush=True)
#     exit(0)


# Install signal handler
# signal.signal(signal.SIGUSR1, signalHandler)
# signal.signal(signal.SIGTERM, SIGTERMHandler)
# print('Signal handler installed', flush=True)
# ========================================================================


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names = list(model_names) + ['resnet18_bottleneck_w64', 'resnet18_bottleneck_w96', 'resnet18_bottleneck_w128',
                                   'resnet50_nobottleneck', 'vit_small', 'vit_small_depth6', 'vit_small_depth3']
models.__dict__.update(resnet_variants.__dict__)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
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
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'], type=str)
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
parser.add_argument('--nnsiam', action='store_true')

# for incremental
parser.add_argument('--incremental_idx_path', default=None, type=str)
parser.add_argument('--strategy', default='iid', type=str)
parser.add_argument('--chunk_size', default=1, type=int)
parser.add_argument('--repeat_batch', default=1, type=int)
parser.add_argument('--n_warmup_steps', default=0, type=int)
parser.add_argument('--steps_per_epoch', default=None, type=int)
parser.add_argument('--n_rounds', default=1, type=int)
parser.add_argument('--fix_all_lr', action='store_true')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

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
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        assert not args.nnsiam
        model = builder.SimSiamViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.dim, args.pred_dim)
    elif args.nnsiam:
        model = builder.NNSiam(partial(models.__dict__[args.arch], zero_init_residual=True), args.dim, args.pred_dim)
    else:
        model = builder.SimSiam(partial(models.__dict__[args.arch], zero_init_residual=True), args.dim, args.pred_dim)
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
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model)  # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        if args.distributed:
            optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False, 'is_encoder': True},
                            {'params': model.module.predictor.parameters(), 'fix_lr': True, 'is_encoder': False}]
        else:
            optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False, 'is_encoder': True},
                            {'params': model.predictor.parameters(), 'fix_lr': True, 'is_encoder': False}]
    else:
        optim_params = model.parameters()

    print(f'Using {args.optimizer} optimizer')
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(optim_params, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), init_lr,
                                      weight_decay=args.weight_decay)
    else:
         raise NotImplementedError

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
        train_stats = train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            wandb.log(train_stats)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch), folder=wandb.run.dir)
            wandb.save(os.path.join(wandb.run.dir, 'checkpoint.pth.tar'))

    if args.multiprocessing_distributed and args.rank == 0:
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, args):
    print('Training, epoch', epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()

    if args.n_warmup_steps > 0 and epoch != 0:
        print('Turning grad off for encoder')
        zero_encoder_lr(optimizer)

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.n_warmup_steps > 0 and args.n_warmup_steps == i and epoch != 0:
            print('Turning grad back on for encoder')
            reset_encoder_lr(optimizer)

        if args.gpu is not None:
            images = [img.cuda(args.gpu, non_blocking=True) for img in images]

        # compute output and loss
        if args.n_augs == 2:
            p1, z1 = model(x1=images[0])
            p2, z2 = model(x1=images[1])

            if args.nnsiam:
                old_z1 = z1
                z1 = model.find_nn(z1)
                z2 = model.find_nn(z2)
                model.update_queue(old_z1)

            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        else:
            loss = fast_simsiam_error(model, images)

        losses.update(loss.item(), images[0].size(0))

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
                loss=losses.avg)


def get_incremental_dataloader(gpu, train_dataset, epoch, total_batch_size, args):
    incremental_idx = np.load(args.incremental_idx_path)
    assert args.chunk_size >= 1  # since we only update the loader every epoch
    assert args.n_rounds >= 1
    epochs_per_round = args.epochs // args.n_rounds
    chunk_i = (epoch % epochs_per_round) // args.chunk_size

    rng = np.random.RandomState(seed=epoch + args.seed)
    if args.strategy == 'accumulate':
        curr_idx = incremental_idx[:len(train_dataset) * (chunk_i + 1) * args.chunk_size // epochs_per_round]
    elif args.strategy == 'current':
        curr_idx = incremental_idx[len(train_dataset) * chunk_i * args.chunk_size // epochs_per_round:
                                   len(train_dataset) * (chunk_i + 1) * args.chunk_size // epochs_per_round]
    elif args.strategy == 'last':
        curr_idx = incremental_idx[-len(train_dataset) * args.chunk_size // epochs_per_round:]
    else:
        raise NotImplementedError

    print("This epoch's dataset size:", len(curr_idx))
    if args.steps_per_epoch is None:
        num_batches = len(train_dataset) // total_batch_size
    else:
        num_batches = args.steps_per_epoch
    idx_lst = []
    batches_so_far = 0
    while batches_so_far < num_batches:
        next_idxs = rng.permutation(curr_idx)[:len(curr_idx) - (len(curr_idx) % total_batch_size)]
        assert len(next_idxs) % total_batch_size == 0
        if args.repeat_batch > 1:
            for i in range(len(next_idxs) // total_batch_size):
                idx_lst.append(np.tile(next_idxs[i * total_batch_size: (i + 1) * total_batch_size], args.repeat_batch))
            batches_so_far += args.repeat_batch * len(next_idxs) // total_batch_size
        else:
            idx_lst.append(next_idxs)
            batches_so_far += len(next_idxs) // total_batch_size
    extended_idx = np.concatenate(idx_lst)[:total_batch_size * num_batches]

    epoch_dataset = Subset(train_dataset, extended_idx)

    if args.distributed:
        # duplicate indices until I have enough
        train_sampler = torch.utils.data.distributed.DistributedSampler(epoch_dataset, shuffle=False)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        epoch_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader, train_sampler


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder=''):
    filename = os.path.join(folder, filename)
    torch.save(state, filename)
    if filename != 'checkpoint.pth.tar':
        shutil.copyfile(filename, os.path.join(folder, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, final_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    if epoch < args.warmup_epochs:
        cur_lr = fix_lr = init_lr * (1 + epoch) / (1 + args.warmup_epochs)
    else:
        cur_lr = final_lr + (init_lr - final_lr) * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        fix_lr = init_lr
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = fix_lr
        else:
            param_group['lr'] = cur_lr


def zero_encoder_lr(optimizer):
    for param_group in optimizer.param_groups:
        if 'is_encoder' in param_group and param_group['is_encoder']:
            assert 'fix_lr' not in param_group or not param_group['fix_lr']
            param_group['original_lr'] = param_group['lr']
            param_group['lr'] = 0


def reset_encoder_lr(optimizer):
    for param_group in optimizer.param_groups:
        if 'is_encoder' in param_group and param_group['is_encoder']:
            assert 'fix_lr' not in param_group or not param_group['fix_lr']
            param_group['lr'] = param_group['original_lr']
            del param_group['original_lr']


if __name__ == '__main__':
    main()
