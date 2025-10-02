from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNetAux(ResNet):
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )
        self.aux_fc = nn.Linear(128 * block.expansion, num_classes)

    def _forward_impl(self, x: Tensor) -> tuple[Tensor, Tensor]:  # 返り値の型を変更
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # 変更開始
        aux = self.avgpool(x)
        aux = torch.flatten(aux, 1)
        aux = self.aux_fc(aux)
        # 変更終了
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, aux

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._forward_impl(x)


def ResNetAux18(num_classes: int = 1000, **kwargs) -> ResNetAux:
    return ResNetAux(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def ResNetAux34(num_classes: int = 1000, **kwargs) -> ResNetAux:
    return ResNetAux(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def ResNetAux50(num_classes: int = 1000, **kwargs) -> ResNetAux:
    return ResNetAux(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def ResNetAux101(num_classes: int = 1000, **kwargs) -> ResNetAux:
    return ResNetAux(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def ResNetAux152(num_classes: int = 1000, **kwargs) -> ResNetAux:
    return ResNetAux(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
