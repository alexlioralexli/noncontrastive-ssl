from torchvision.models.resnet import _resnet, Bottleneck, ResNet, BasicBlock
from typing import Any


def resnet18_bottleneck_w64(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('none', Bottleneck, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_bottleneck_w96(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 96
    return _resnet('none', Bottleneck, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_bottleneck_w128(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 128
    return _resnet('none', Bottleneck, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50_nobottleneck(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50_nobottleneck', BasicBlock, [5, 6, 9, 4], pretrained, progress,
                   **kwargs)
