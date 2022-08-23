import os
import torch
import autolearn.simsiam as simsiam
from torchvision import transforms
from autolearn.dataset_with_path import ImageDatasetWithPath


def simsiam_train_augs():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    return augmentation


def get_imagenet_dataset(args):
    traindir = os.path.join(args.data, 'train')
    augmentation = simsiam_train_augs()

    train_dataset = ImageDatasetWithPath(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    return train_dataset


def get_imagenet(args):
    train_dataset = get_imagenet_dataset(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_sampler, train_loader


def get_candidate_dataset(itr_data_folder, n):
    augmentation = simsiam_train_augs()
    candidate_dataset = ImageDatasetWithPath(
        itr_data_folder,
        simsiam.loader.NCropsTransform(transforms.Compose(augmentation), n=n),
        is_valid_file=is_valid_file
    )
    return candidate_dataset


def get_candidate_dataloader(itr_data_folder, n, args, val=False):
    candidate_dataset = get_candidate_dataset(itr_data_folder, n)
    if args.distributed:
        candidate_sampler = torch.utils.data.distributed.DistributedSampler(candidate_dataset)
    else:
        candidate_sampler = None
    candidate_loader = torch.utils.data.DataLoader(
        candidate_dataset, batch_size=args.batch_size, shuffle=(candidate_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=candidate_sampler, drop_last=not val)

    return candidate_loader


def is_valid_file(fpath):
    return ('jpg' == fpath[-3:]) or ('png' == fpath[-3:])
