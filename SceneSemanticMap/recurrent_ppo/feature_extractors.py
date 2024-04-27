from enum import Enum
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from typing import Tuple

class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5


_init_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
    Initialization.Normal: torch.nn.init.normal_,
}


class Swish(torch.nn.Module):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mul(data, torch.sigmoid(data))

class ResNetBlock(nn.Module):
    def __init__(self, channel: int):
        """
        Creates a ResNet Block.
        :param channel: The number of channels in the input (and output) tensors of the
        convolutions
        """
        super().__init__()
        self.layers = nn.Sequential(
            Swish(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
            Swish(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.layers(input_tensor)


class ResNetVisualEncoder(nn.Module):
    def __init__(
            self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        layers = []
        last_channel = initial_channels
        for _, channel in enumerate(n_channels):
            layers.append(nn.Conv2d(last_channel, channel, [3, 3], [1, 1], padding=1))
            layers.append(nn.MaxPool2d([3, 3], [2, 2]))
            height, width = pool_out_shape((height, width), 3)
            for _ in range(n_blocks):
                layers.append(ResNetBlock(channel))
            last_channel = channel
        layers.append(Swish())

        self.final_flat_size = n_channels[-1] * height * width
        self.dense = linear_layer(
            self.final_flat_size,
            output_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.41,  # Use ReLU gain
        )
        self.sequential = nn.Sequential(*layers)

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # if visual_obs.shape[1]>12:
        #    visual_obs = visual_obs.permute([0, 3, 1, 2])

        hidden = self.sequential(visual_obs)
        before_out = hidden.reshape(-1, self.final_flat_size)
        return torch.relu(self.dense(before_out))


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a max pooling layer.
    kernel_size corresponds to the inputs of the
    torch.nn.MaxPool2d layer (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    :param kernel_size: The size of the kernel of the convolution
    """
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


def linear_layer(
        input_size: int,
        output_size: int,
        kernel_init: Initialization = Initialization.XavierGlorotUniform,
        kernel_gain: float = 1.0,
        bias_init: Initialization = Initialization.Zero,
) -> torch.nn.Module:
    """
    Creates a torch.nn.Linear module and initializes its weights.
    :param input_size: The size of the input tensor
    :param output_size: The size of the output tensor
    :param kernel_init: The Initialization to use for the weights of the layer
    :param kernel_gain: The multiplier for the weights of the kernel. Note that in
    TensorFlow, the gain is square-rooted. Therefore calling  with scale 0.01 is equivalent to calling
        KaimingHeNormal with kernel_gain of 0.1
    :param bias_init: The Initialization to use for the weights of the bias layer
    """
    layer = torch.nn.Linear(input_size, output_size)
    if (
            kernel_init == Initialization.KaimingHeNormal
            or kernel_init == Initialization.KaimingHeUniform
    ):
        _init_methods[kernel_init](layer.weight.data, nonlinearity="linear")
    else:
        _init_methods[kernel_init](layer.weight.data)
    layer.weight.data *= kernel_gain
    _init_methods[bias_init](layer.bias.data)
    return layer