import math

import gymnasium
import numpy as np
from gymnasium import spaces
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.custom_networks import ResNetVisualEncoder, TensorFusion


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self,observation_space:spaces.Box,features_dim:int=256):
        super().__init__(observation_space,features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels,32,kernel_size=8,stride=4,padding=0),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten,features_dim),nn.ReLU())

    def forward(self,observations:torch.Tensor):
        return self.linear(self.cnn(observations))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Dict,use_resnet:bool):
        super().__init__(observation_spaces,features_dim=1)
        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_spaces.spaces.items():
            if key=="image" or key=="semantic" or key=="depth" or key=="gray" or key=="semanticlidar" or key == "infrared" :
                if(use_resnet):
                    extractors[key] = ResNetVisualEncoder(subspace.shape[1],subspace.shape[2],subspace.shape[0],256)
                else:
                    extractors[key] = nn.Sequential(
                        nn.Conv2d(subspace.shape[0], 32, kernel_size=8, stride=4, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                with torch.no_grad():
                    n_flatten = extractors[key](torch.as_tensor(subspace.sample()[None]).float()).shape[1]
                    total_concat_size += n_flatten
            elif key=="ray" or key=="label" or key=="vec" :
                extractors[key] = nn.Linear(subspace.shape[0],256)
                total_concat_size += 256
            elif key=="objectDet" or key=="speed" or key=="intention" or key=="state":
                total_concat_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self,observations):
        encoded_tensor_list = []
        for key,obs in observations.items():
            if key in set(it[0] for it in self.extractors.items()):
                encoded_tensor_list.append(self.extractors[key](observations[key]))
            else:
                encoded_tensor_list.append(observations[key])
        '''for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))'''
        return torch.cat(encoded_tensor_list,dim=1)

class ResNetImageExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Box,output_shape):
        super().__init__(observation_spaces,features_dim=output_shape)

        n_inut_channels = observation_spaces.shape[0]

        self.resnet = ResNetVisualEncoder(observation_spaces.shape[1],observation_spaces.shape[2],n_inut_channels,output_shape)

    def forward(self,observations):
        return self.resnet(observations)


class TensorFusionExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Dict,ckpt):
        super().__init__(observation_spaces,features_dim=1)
        self.tensorfusion = TensorFusion(observation_spaces)
        self.check_point = torch.load(ckpt)
        self.tensorfusion.load_state_dict(self.check_point['state_dict'])
        self.tensorfusion.eval()
        for param in self.tensorfusion.parameters():
            param.requires_grad = False

        self._features_dim = int(self.tensorfusion.out_dim)
    def forward(self,observations):
        return self.tensorfusion(observations)
    def predict(self,obs1,obs2):
        return self.tensorfusion.predict(obs1,obs2)

class LowRankTensorFusionExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Dict,output_dim,rank,ckpt):
        super().__init__(observation_spaces,features_dim=output_dim)
        self.lowranktensorfusion = LowRankTensorFusion(observation_spaces,output_dim,rank)
        self._features_dim = output_dim
        self.check_point = torch.load(ckpt)
        self.lowranktensorfusion.load_state_dict(self.check_point['state_dict'])
        self.lowranktensorfusion.eval()
        for param in self.lowranktensorfusion.parameters():
            param.requires_grad = False
    def forward(self,observations):
        return self.lowranktensorfusion(observations)
    def predict(self,obs1,obs2):
        return self.lowranktensorfusion.predict(obs1,obs2)

class TranformerFusionExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Dict,hyper_param,ckpt):
        super().__init__(observation_spaces,features_dim=hyper_param.output_dim)
        self.MULTModel = MULTModel(observation_spaces,hyper_params=hyper_param)
        self.check_point = torch.load(ckpt)
        self.MULTModel.load_state_dict(self.check_point['state_dict'])
        self.MULTModel.eval()
        for param in self.MULTModel.parameters():
            param.requires_grad = False
    def forward(self,observations):
        return self.MULTModel(observations)
    def predict(self,obs1,obs2):
        return self.MULTModel.predict(obs1,obs2)


if __name__ == '__main__':
    obs_1 = spaces.Box(low=0, high=255, shape=(12, 84,84), dtype=np.uint8)
    obs_2 = spaces.Box(low=-math.inf, high=math.inf, shape=(1600,), dtype=np.float32)

    # obs_shape = gymnasium.spaces.Tuple((obs_1,obs_2),seed=42)
    obs_shape = gymnasium.spaces.Dict(spaces={"image": obs_1, "ray": obs_2}, seed=42)
    model = TensorFusion(obs_shape).to("cuda")
    obs1 = torch.ones(size=(1,12,84,84)).to("cuda")
    obs2 = torch.ones(size=(1,1600)).to("cuda")

    obs = {"image":obs1,"ray":obs2}
    print(model(obs).shape)
