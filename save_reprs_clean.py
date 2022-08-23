"""Given some pretrained model, get representations for every image in a desired dataset"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import torchvision
import torchvision.models as torchvision_models
from torchvision import datasets, transforms
from tqdm import tqdm
from autolearn.simsiam.data import imagenet_val, imagenet_train_with_val_augs

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Saving MoCo representations')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture')
parser.add_argument('--dataset_type', default='imagenet_train', type=str)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--pretrained_distributed', action='store_true',
                    help='was the pretrained checkpoint distributed?')
parser.add_argument('--exp_id', default=-1, type=int)
parser.add_argument('--extra_info', default=None, type=str, help='extra info to put in folder name')
parser.add_argument('--alg', type=str, default='simsiam')

def create_and_load_model(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    # device = f'cuda:{args.gpu}'
    # model, preprocess = clip.load(args.arch, device)
    model = torchvision_models.__dict__[args.arch]()
    args.repr_size = model.fc.weight.shape[1]

    # load from checkpoint
    assert args.pretrained
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename pre-trained keys
        state_dict = checkpoint['state_dict']

        if args.alg in {'moco', 'byol'}:
            encoder_str = 'base_encoder'
        elif args.alg == 'distill':
            encoder_str = ''
        else:
            encoder_str = 'encoder'
        if args.pretrained_distributed:  # if the simsiam training was with distributed
            key_prefix = f'module.{encoder_str}'
        else:
            key_prefix = encoder_str

        if args.alg == 'distill':
            for k in list(state_dict.keys()):
                if k.startswith('fc'):
                    del state_dict[k]
        else:
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith(key_prefix) and not k.startswith(key_prefix + 'fc'):
                    # remove prefix
                    state_dict[k[len(key_prefix + "."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.pretrained))

    model.fc = nn.Identity()
    return model.cuda(args.gpu)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print("Use GPU: {} for training".format(args.gpu))

    model = create_and_load_model(args)
    cudnn.benchmark = True

    # Load the dataset
    if args.dataset_type == 'imagenet_train':
        dataset = imagenet_train_with_val_augs(args)
    elif args.dataset_type == 'imagenet_val':
        dataset = imagenet_val(args)
    else:
        raise NotImplementedError
    print("Dataset size:", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True, drop_last=False)
    dataset_name, split = args.dataset_type.split('_')
    folder = f"data/{args.arch}_exp{args.exp_id:04d}"
    if args.extra_info:
        folder += f"_{args.extra_info}"
    folder += f"/{dataset_name}"

    os.makedirs(folder, exist_ok=True)
    for epoch in range(args.epochs):
        print('Computing representations')
        result = get_reprs(loader, model, args, len(dataset))
        fname = f"{split}_reprs_e{epoch}.pt"
        print('Saving representations')
        torch.save(dict(representations=result['representations'],
                        targets=result['targets']),
                   os.path.join(folder, fname))
        del result  # clearing memory


def get_reprs(train_loader, model, args, dataset_len):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    representations = torch.zeros(dataset_len, args.repr_size, dtype=torch.float32, device='cpu')
    targets = torch.zeros(dataset_len)

    with torch.inference_mode():
        for i, (images, target) in enumerate(tqdm(train_loader)):
            images = images.cuda(args.gpu, non_blocking=True)
            output = model(images)
            representations[i * args.batch_size: i * args.batch_size + len(output)] = output.detach().cpu()
            targets[i * args.batch_size: i * args.batch_size + len(output)] = target.detach()

    return dict(representations=representations, targets=targets)


if __name__ == '__main__':
    main()
