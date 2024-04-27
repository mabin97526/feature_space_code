import math
from enum import Enum
from typing import Tuple
import gymnasium
import numpy as np
from gymnasium import spaces
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from stable_baselines3.common.distributions import *

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

class ImageEncoder(nn.Module):
    def __init__(self,inputchannel,features_dim,subspace):
        super(ImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(inputchannel, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).to("cuda")
        #84*84*12
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(subspace.sample()[None]).float().to("cuda")).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten,features_dim),nn.ReLU()).to("cuda")
    def forward(self,s):

        return self.linear(self.cnn(s))

class RayEncoder(nn.Module):
    def __init__(self,inputchannel,features_dim,subspace):
        super(RayEncoder, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(inputchannel, 16, 1, 4,padding=0),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1, 2,padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).to("cuda")
        with torch.no_grad():
            test1 = torch.as_tensor(subspace.sample()[None]).float()
            test1 = test1.reshape(test1.shape[-2],inputchannel,1).to("cuda")
            #print(inputchannel)
            n_flatten = self.extractor(test1).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten,features_dim),nn.ReLU()).to("cuda")
    def forward(self,s):
        if(len(s.shape)==2):
            s = torch.unsqueeze(s,dim=2)
        return self.linear(self.extractor(s))


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


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        flattened = []

        for modality in modalities.values():
            flattened.append(torch.flatten(modality, start_dim=0))
        return torch.cat(flattened, dim=0)


class TensorFusion(nn.Module):
    def __init__(self, observation_space):
        super(TensorFusion, self).__init__()
        self.extractors = {}
        for key, subspace in observation_space.spaces.items():
            if key == "image" or key == "semantic" or key == "depth" or key == "gray":
                self.extractors[key] = ImageEncoder(subspace.shape[0], 256, subspace)
            elif key == "ray_conv1d":
                self.extractors[key] = RayEncoder(subspace.shape[0], 256, subspace)
            elif key == "ray":
                self.extractors[key] = nn.Linear(subspace.shape[0],16).to("cuda")
        #modalNums = len(self.extractors)
        self.action_dist = SquashedDiagGaussianDistribution(2)  # type: ignore[assignment]

        self.out_dim = 257*17
        self.mu = nn.Linear(self.out_dim*2, 2).to("cuda")
        self.log_std = nn.Linear(self.out_dim*2, 2).to("cuda")  # type: ignore[assignment]

    def forward(self, modalities):

        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(modalities[key]))

        mod0 = encoded_tensor_list[0]
        nonfeature_size = mod0.shape[:-1]
        m = torch.cat(
            (Variable(torch.ones(*nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0),
            dim=-1)
        for mod in encoded_tensor_list[1:]:
            mod = torch.cat(
                (Variable(torch.ones(*nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod),
                dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m

    def predict(self,s,s_):
        s = self.forward(s)
        s_ = self.forward(s_)
        s_ = torch.cat((s,s_),dim=-1)

        mean_actions = self.mu(s_)
        log_std = self.log_std(s_)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=True)


class LowRankTensorFusion(nn.Module):
    def __init__(self,observation_space,output_dim,rank,flatten=True):
        super(LowRankTensorFusion, self).__init__()
        self.observation_space = observation_space
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten
        self.extractors = {}
        for key, subspace in observation_space.spaces.items():
            if key == "image" or key == "semantic" or key == "depth" or key == "gray":
                self.extractors[key] = ImageEncoder(subspace.shape[0], 256, subspace)
            elif key == "rayconv":
                self.extractors[key] = RayEncoder(subspace.shape[0], 256, subspace)
            elif key == "ray":
                self.extractors[key] = nn.Linear(subspace.shape[0],16).to("cuda")
        self.factors = []
        extractor_len = len(self.extractors)
        '''for i in range(extractor_len):
            factor = nn.Parameter(torch.Tensor(
                self.rank,257,self.output_dim
            )).to(torch.device("cuda:0"))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)'''
        factor1 = nn.Parameter(torch.Tensor(self.rank,257,self.output_dim)).to(torch.device("cuda:0"))
        nn.init.xavier_normal(factor1)
        self.factors.append(factor1)
        factor2 = nn.Parameter(torch.Tensor(self.rank,17,self.output_dim)).to("cuda")
        nn.init.xavier_normal(factor2)
        self.factors.append(factor2)
        self.fusion_weights = nn.Parameter(torch.Tensor(1,self.rank)).to(torch.device("cuda:0"))
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1,self.output_dim)
        ).to(torch.device("cuda:0"))
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        self.action_dist = SquashedDiagGaussianDistribution(2)
        self.mu = nn.Linear(self.output_dim*2, 2).to("cuda")
        self.log_std = nn.Linear(self.output_dim*2, 2).to("cuda")  # type: ignore[assignment]
    def forward(self,modalities):

        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            #print(modalities[key])
            encoded_tensor_list.append(extractor(modalities[key]))
        batch_size = encoded_tensor_list[0].shape[0]
        fused_tensor = 1
        for (modality,factor) in zip(encoded_tensor_list,self.factors):
            ones = Variable(torch.ones(batch_size,1).type(modality.dtype),requires_grad=False).to(torch.device("cuda:0"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones,torch.flatten(modality,start_dim=1)),dim=1
                )
            else:
                modality_withones = torch.cat((ones,modality),dim=1)
            modality_factor = torch.matmul(modality_withones,factor)
            fused_tensor = fused_tensor * modality_factor
        output = torch.matmul(self.fusion_weights,fused_tensor.permute(
            1,0,2
        )).squeeze() + self.fusion_bias
        output = output.view(-1,self.output_dim)
        return output

    def predict(self,s,s_):
        s = self.forward(s)
        s_ = self.forward(s_)
        s_ = torch.cat((s,s_),dim=-1)

        mean_actions = self.mu(s_)
        log_std = self.log_std(s_)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=True)

class MULTModel(nn.Module):
    class DefaultHyperParams():
        num_heads=3
        layers = 3
        attn_dropout = 0.1
        attn_dropout_modalities = [0.0] * 1000
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.0
        embed_dropout = 0.25
        embed_dim = 9
        attn_mask = True
        output_dim = 1
        all_steps = False

    def __init__(self,observation_space,hyper_params=DefaultHyperParams):
        super().__init__()
        self.observation_space = observation_space
        self.extractors = {}
        for key, subspace in observation_space.spaces.items():
            if key == "image" or key == "semantic" or key == "depth" or key == "gray":
                self.extractors[key] = ImageEncoder(subspace.shape[0], 256, subspace)
            elif key == "rayconv":
                self.extractors[key] = RayEncoder(subspace.shape[0], 256, subspace)
            elif key == "ray":
                self.extractors[key] = nn.Linear(subspace.shape[0], 256).to("cuda")
        self.n_modalities = len(self.extractors.keys())
        self.embed_dim = hyper_params.embed_dim
        self.num_heads = hyper_params.num_heads
        self.layers = hyper_params.layers
        self.attn_dropout = hyper_params.attn_dropout
        self.attn_dropout_modalities = hyper_params.attn_dropout_modalities
        self.relu_dropout = hyper_params.relu_dropout
        self.res_dropout = hyper_params.res_dropout
        self.out_dropout = hyper_params.out_dropout
        self.embed_dropout = hyper_params.embed_dropout
        self.attn_mask = hyper_params.attn_mask
        self.all_steps = hyper_params.all_steps

        combined_dim = self.embed_dim * self.n_modalities *self.n_modalities

        output_dim = hyper_params.output_dim
        self.proj = [nn.Conv1d(256,self.embed_dim,kernel_size=1,padding=0,bias=False) for i in range(self.n_modalities)]
        self.proj = nn.ModuleList(self.proj)

        # crossmodal attention
        self.trans = [nn.ModuleList([self.get_network(i,j,mem=False) for j in range(self.n_modalities)]) for i in range(self.n_modalities)]
        self.trans = nn.ModuleList(self.trans)

        # self attn
        self.trans_mems = [self.get_network(
            i,i,mem=True,layers=3
        ) for i in range(self.n_modalities)]
        self.trans_mems = nn.ModuleList(self.trans_mems)

        self.proj1 = nn.Linear(combined_dim,combined_dim)
        self.proj2 = nn.Linear(combined_dim,combined_dim)
        self.out_layer = nn.Linear(combined_dim,output_dim)

        self.action_dist = SquashedDiagGaussianDistribution(2)
        self.mu = nn.Linear(output_dim*2, 2).to("cuda")
        self.log_std = nn.Linear(output_dim*2, 2).to("cuda")  # type: ignore[assignment]


    def get_network(self,mod1,mod2,mem,layers=-1):
        if not mem:
            embed_dim = self.embed_dim
            attn_dropout = self.attn_dropout_modalities[mod2]
        else:
            embed_dim = self.n_modalities *self.embed_dim
            attn_dropout = self.attn_dropout
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers = max(self.layers,layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask
                                  )

    def forward(self,modalities):
        encoded_tensor_list = []
        batch_size = modalities["image"].shape[0]
        for key, extractor in self.extractors.items():
            modal_feature = torch.unsqueeze(extractor(modalities[key]),-1) ## batch_size * 256 --> batch_size * 256 *1
            encoded_tensor_list.append(modal_feature)
        proj_x = [self.proj[i](encoded_tensor_list[i]) for i in range(self.n_modalities)]
        proj_x = torch.stack(proj_x)
        proj_x = proj_x.permute(0,3,1,2)

        hs = []
        last_hs = []
        for i in range(self.n_modalities):
            h = []
            for j in range(self.n_modalities):
                h.append(self.trans[i][j](proj_x[i],proj_x[j],proj_x[j]))
            h = torch.cat(h,dim=2)
            if self.all_steps:
                hs.append(h)
            else:
                last_hs.append(h[-1])
        if self.all_steps:
            out = torch.cat(hs,dim=2)
            out = out.permute(1,0,2)
        else:
            out =  torch.cat(last_hs,dim=1)

        out_proj = self.proj2(
            F.dropout(F.relu(self.proj1(out)),p=self.out_dropout,training=self.training)
        )
        out_proj += out
        out = self.out_layer(out_proj)
        return out.reshape(batch_size,-1)
    def predict(self,s,s_):
        s = self.forward(s)
        s_ = self.forward(s_)
        s_ = torch.cat((s,s_),dim=-1)

        mean_actions = self.mu(s_)
        log_std = self.log_std(s_)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=True)



class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers.

    Each layer is a :class:`TransformerEncoderLayer`.

    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        """Initialize Transformer Encoder.
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of heads
            layers (int): Number of layers
            attn_dropout (float, optional): Probability of dropout in attention mechanism. Defaults to 0.0.
            relu_dropout (float, optional): Probability of dropout after ReLU. Defaults to 0.0.
            res_dropout (float, optional): Probability of dropout in residual layer. Defaults to 0.0.
            embed_dropout (float, optional): Probability of dropout in embedding layer. Defaults to 0.0.
            attn_mask (bool, optional): Whether to apply a mask to the attention or not. Defaults to False.
        """
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Apply Transformer Encoder to layer input.

        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            # Add positional embedding
            x += self.embed_positions(x_in.transpose(0, 1)
                                      [:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                # Add positional embedding
                x_k += self.embed_positions(x_in_k.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
                # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Implements encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """Instantiate TransformerEncoderLayer Module.
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int, optional): Number of heads. Defaults to 4.
            attn_dropout (float, optional): Dropout for attention mechanism. Defaults to 0.1.
            relu_dropout (float, optional): Dropout after ReLU. Defaults to 0.1.
            res_dropout (float, optional): Dropout after residual layer. Defaults to 0.1.
            attn_mask (bool, optional): Whether to apply an attention mask or not. Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # The "Add & Norm" part in the paper
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Apply TransformerEncoderLayer to Layer Input.

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self._maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self._maybe_layer_norm(0, x_k, before=True)
            x_v = self._maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(0, x, after=True)

        residual = x
        x = self._maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(1, x, after=True)
        return x

    def _maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    """Generate buffered future mask.
    Args:
        tensor (torch.Tensor): Tensor to initialize mask from.
        tensor2 (torch.Tensor, optional): Tensor to initialize target mask from. Defaults to None.
    Returns:
        torch.Tensor: Buffered future mask.
    """
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(
        torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    """Generate Linear Layer with given parameters and Xavier initialization.
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool, optional): Whether to include a bias term or not. Defaults to True.
    Returns:
        nn.Module: Initialized Linear Module.
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    """Generate LayerNorm Layer with given parameters.
    Args:
        embedding_dim (int): Embedding dimension
    Returns:
        nn.Module: Initialized LayerNorm Module
    """
    m = nn.LayerNorm(embedding_dim)
    return m


"""Implements Positional Encoding.
Adapted from fairseq repo.
"""


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).

    Args:
        tensor (torch.Tensor): Tensor to generate padding on.
        padding_idx (int): Position numbers start at padding_idx + 1
        left_pad (bool): Whether to pad from the left or from the right.
    Returns:
        torch.Tensor: Padded output
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(
        make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos,
                     out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[
                :tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - \
                    mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0):
        """Instantiate SinusoidalPositionalEmbedding Module.
        Args:
            embedding_dim (int): Embedding dimension
            padding_idx (int, optional): Padding index. Defaults to 0.
            left_pad (int, optional): Whether to pad from the left or not. Defaults to 0.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        # device --> actual weight; due to nn.DataParallel :-(
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Apply PositionalEncodings to Input.

        Input is expected to be of size [bsz x seqlen].
        Args:
            input (torch.Tensor): Layer input
        Returns:
            torch.Tensor: Layer output
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape((bsz, seq_len, -1)).detach()

if __name__ == '__main__':
    obs_1 = spaces.Box(low=0, high=255, shape=(12, 84, 84), dtype=np.uint8)
    obs_2 = spaces.Box(low=-math.inf, high=math.inf, shape=(1600,), dtype=np.float32)

    # obs_shape = gymnasium.spaces.Tuple((obs_1,obs_2),seed=42)
    obs_shape = gymnasium.spaces.Dict(spaces={"image": obs_1, "ray": obs_2}, seed=42)

    class HyperParams(MULTModel.DefaultHyperParams):
        num_heads = 2
        embed_dim = 256
        output_dim = 64
        all_steps = True
    obs1 = torch.ones((10, 12, 84, 84), dtype=torch.float32).to("cuda")
    obs2 = torch.ones((10, 1600), dtype=torch.float32).to("cuda")
    obs = {"image": obs1, "ray": obs2}
    obs3 = torch.zeros((10,12,84,84),dtype=torch.float32).to("cuda")
    obs4 = torch.ones((10,1600),dtype=torch.float32).to("cuda")
    obs_ = {"image":obs3,"ray":obs4}
    #model = LowRankTensorFusion(obs_shape,output_dim=256,rank=64)
    model = MULTModel(obs_shape,HyperParams).to("cuda")
    y = model.predict(obs,obs_)
    print(y)
    y_ = torch.zeros((10,2)).to("cuda")
    loss = F.mse_loss(y,y_)
    print(loss)