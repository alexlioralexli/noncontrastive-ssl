"""
Contains all the data loaders that we need
"""
import os
import autolearn.simsiam.loader as loader
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
from typing import Any, Tuple
from autolearn.dataset_with_path import ImageDatasetWithIndex, ImageDatasetWithPath

DATASET_FOLDER = os.getenv('DATASET_FOLDER')
if DATASET_FOLDER is None:
    DATASET_FOLDER = os.path.expanduser('~/datasets')


def simsiam_train_aug():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    return augmentation


def simsiam_val_aug():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    return augmentation


def lincls_aug():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]


def imagenet_train(args, n_augs=2):
    if n_augs < 2:
        assert n_augs == 1
        print('Warning: only using 1 augmentation')

    # create imagefolder dataset
    traindir = os.path.join(args.data, 'train')
    augmentation = simsiam_train_aug()

    if n_augs == 2:
        data_transform = loader.TwoCropsTransform(transforms.Compose(augmentation))
    else:
        data_transform = loader.NCropsTransform(transforms.Compose(augmentation), n=n_augs)
    train_dataset = datasets.ImageFolder(traindir, data_transform)

    return train_dataset


def imagenet_train_subset(args, n_augs=2):
    assert args.idx_path is not None
    idx = np.load(args.idx_path)

    full_train_set = imagenet_train(args, n_augs=n_augs)
    train_subset = Subset(full_train_set, idx)
    print(len(full_train_set), len(train_subset))

    return train_subset


def imagenet_train_with_idx(args, n_augs=2):
    # create imagefolder dataset
    traindir = os.path.join(args.data, 'train')

    augmentation = simsiam_train_aug()
    if n_augs == 2:
        data_transform = loader.TwoCropsTransform(transforms.Compose(augmentation))
    else:
        data_transform = loader.NCropsTransform(transforms.Compose(augmentation), n=n_augs)
    train_dataset = ImageDatasetWithIndex(traindir, data_transform)

    print('using n_augs=', n_augs)
    return train_dataset


def imagenet_lincls_train(args):
    train_transform = transforms.Compose(lincls_aug())
    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(traindir, train_transform)

    return train_dataset


def imagenet_lincls_train_subset(args):
    assert args.idx_path is not None
    idx = np.load(args.idx_path)
    full_train_set = imagenet_lincls_train(args)
    train_subset = Subset(full_train_set, idx)
    print(len(full_train_set), len(train_subset))

    return train_subset


def imagenet_train_with_val_augs(args):
    train_transform = transforms.Compose(simsiam_val_aug())
    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(traindir, train_transform)

    return train_dataset


def imagenet_val(args):
    valdir = os.path.join(args.data, 'val')
    augmentation = transforms.Compose(simsiam_val_aug())
    val_dataset = datasets.ImageFolder(valdir, augmentation)

    return val_dataset


def imagenet_val_with_path(args):
    valdir = os.path.join(args.data, 'val')
    augmentation = transforms.Compose(simsiam_val_aug())
    val_dataset = ImageDatasetWithPath(valdir, augmentation)

    return val_dataset


def imagenet_val_with_train_augs_and_path(args, n_augs=2):
    assert n_augs >= 2

    # create imagefolder dataset
    valdir = os.path.join(args.data, 'val')
    augmentation = simsiam_train_aug()
    if n_augs == 2:
        data_transform = loader.TwoCropsTransform(transforms.Compose(augmentation))
    else:
        data_transform = loader.NCropsTransform(transforms.Compose(augmentation), n=n_augs)
    val_dataset = ImageDatasetWithPath(valdir, data_transform)

    return val_dataset


def places_simsiam_train(args):
    augmentation = simsiam_train_aug()
    train_dataset = datasets.places365.Places365(
        root=os.path.join(DATASET_FOLDER, 'places365'),
        split='train-standard',
        small=args.small,
        download=False,
        transform=loader.TwoCropsTransform(transforms.Compose(augmentation))
    )

    return train_dataset


def places_train_subset(args):
    assert args.idx_path is not None
    idx = np.load(args.idx_path)

    full_train_set = places_simsiam_train(args)
    train_subset = Subset(full_train_set, idx)
    print(len(full_train_set), len(train_subset))

    return train_subset


def places_lincls_train(args):
    train_transform = transforms.Compose(lincls_aug())
    train_dataset = datasets.places365.Places365(root=os.path.join(DATASET_FOLDER, 'places365'), split='train-standard',
                                                 small=args.small, download=False, transform=train_transform)

    return train_dataset


def places_lincls_train_subset(args):
    assert args.idx_path is not None
    idx = np.load(args.idx_path)
    full_train_set = places_lincls_train(args)
    train_subset = Subset(full_train_set, idx)
    print(len(full_train_set), len(train_subset))

    return train_subset


def places_val(args):
    val_transform = transforms.Compose(simsiam_val_aug())
    val_dataset = datasets.places365.Places365(root=os.path.join(DATASET_FOLDER, 'places365'), split='val',
                                               small=args.small, download=False, transform=val_transform)

    return val_dataset


def places_val_with_train_aug_and_path(args, n_augs=2):
    augmentation = simsiam_train_aug()
    if n_augs == 2:
        data_transform = loader.TwoCropsTransform(transforms.Compose(augmentation))
    else:
        data_transform = loader.NCropsTransform(transforms.Compose(augmentation), n=n_augs)
    val_dataset = PlacesWithFile(
        root=os.path.join(DATASET_FOLDER, 'places365'),
        split='val',
        small=args.small,
        download=False,
        transform=data_transform
    )

    return val_dataset


def places_val_with_path(args):
    augmentation = simsiam_val_aug()
    val_dataset = PlacesWithFile(
        root=os.path.join(DATASET_FOLDER, 'places365'),
        split='val',
        small=args.small,
        download=False,
        transform=transforms.Compose(augmentation)
    )

    return val_dataset


class PlacesWithFile(datasets.places365.Places365):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        file, target = self.imgs[index]
        image = self.loader(file)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, file
