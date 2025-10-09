from typing import Union

import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv1x1


class PlainBasicBlock(BasicBlock):
    def __init__(self, use_bn: bool = False, use_skip: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_skip = use_skip

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)  # use_bn=Falseの時はIdentityになっている
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # use_bn=Falseの時はIdentityになっている

        if self.use_skip:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)

        return out


class PlainBottleneck(Bottleneck):
    def __init__(self, use_bn: bool = False, use_skip: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_skip = use_skip

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)  # use_bn=Falseの時はIdentityになっている
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # use_bn=Falseの時はIdentityになっている
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)  # use_bn=Falseの時はIdentityになっている

        if self.use_skip:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)

        return out


class Plain(ResNet):
    def __init__(
        self,
        block: type[Union[PlainBasicBlock, PlainBottleneck]],
        use_bn: bool = False,
        use_skip: bool = False,
        use_aux: bool = False,
        num_classes: int = 1000,
        *args,
        **kwargs,
    ):
        self.use_skip = use_skip
        self.block_class = block

        # use_bn=Falseの時はBNの代わりにIdentityを使う
        if not use_bn:
            kwargs["norm_layer"] = nn.Identity

        super().__init__(block=block, num_classes=num_classes, *args, **kwargs)
        self.use_aux = use_aux
        if use_aux:
            self.aux_fc = nn.Linear(128 * self.block_class.expansion, num_classes)

    def _make_layer(
        self,
        block: type[Union[PlainBasicBlock, PlainBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                use_skip=self.use_skip,
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    use_skip=self.use_skip,
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)  # use_bn=Falseの時はIdentityになっている
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        if self.use_aux:
            aux = self.avgpool(x)
            aux = torch.flatten(aux, 1)
            aux = self.aux_fc(aux)
        else:
            aux = None

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, aux

    def forward(self, x: Tensor):
        return self._forward_impl(x)


def plain18(
    use_bn: bool = False, use_skip: bool = False, use_aux: bool = False, **kwargs
):
    return Plain(
        block=PlainBasicBlock,
        layers=[2, 2, 2, 2],
        use_bn=use_bn,
        use_skip=use_skip,
        use_aux=use_aux,
        **kwargs,
    )


def plain34(
    use_bn: bool = False, use_skip: bool = False, use_aux: bool = False, **kwargs
):
    return Plain(
        block=PlainBasicBlock,
        layers=[3, 4, 6, 3],
        use_bn=use_bn,
        use_skip=use_skip,
        use_aux=use_aux,
        **kwargs,
    )


def plain50(
    use_bn: bool = False, use_skip: bool = False, use_aux: bool = False, **kwargs
):
    return Plain(
        block=PlainBottleneck,
        layers=[3, 4, 6, 3],
        use_bn=use_bn,
        use_skip=use_skip,
        use_aux=use_aux,
        **kwargs,
    )


def plain101(
    use_bn: bool = False, use_skip: bool = False, use_aux: bool = False, **kwargs
):
    return Plain(
        block=PlainBottleneck,
        layers=[3, 4, 23, 3],
        use_bn=use_bn,
        use_skip=use_skip,
        use_aux=use_aux,
        **kwargs,
    )


def plain152(
    use_bn: bool = False, use_skip: bool = False, use_aux: bool = False, **kwargs
):
    return Plain(
        block=PlainBottleneck,
        layers=[3, 8, 36, 3],
        use_bn=use_bn,
        use_skip=use_skip,
        use_aux=use_aux,
        **kwargs,
    )
