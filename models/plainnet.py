from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import conv1x1, conv3x3,BasicBlock, Bottleneck, ResNet
from torchvision.utils import _log_api_usage_once


class PlainBasicBlock(BasicBlock):
    def forward(self, x: Tensor) -> Tensor:
        # identity = x　# 変更　コメントアウト

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 変更　下４行コメントアウト
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out

class PlainBottleneck(Bottleneck):
    def forward(self, x: Tensor) -> Tensor:
        # identity = x  # 変更　コメントアウト

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 変更　下４行コメントアウト
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out

class PlainNet(ResNet):
    """Plain network built on ResNet backbone without residual shortcuts."""

    def __init__(
        self,
        block: type[Union[PlainBasicBlock, PlainBottleneck]] = PlainBasicBlock,
        layers: list[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            **kwargs,
        )


def PlainNet18(num_classes: int = 1000, **kwargs) -> PlainNet:
    return PlainNet(PlainBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def PlainNet34(num_classes: int = 1000, **kwargs) -> PlainNet:
    return PlainNet(PlainBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def PlainNet50(num_classes: int = 1000, **kwargs) -> PlainNet:
    return PlainNet(PlainBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def PlainNet101(num_classes: int = 1000, **kwargs) -> PlainNet:
    return PlainNet(PlainBottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def PlainNet152(num_classes: int = 1000, **kwargs) -> PlainNet:
    return PlainNet(PlainBottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
