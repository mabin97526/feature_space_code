import copy
import math
from functools import wraps

import torch as th
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.optim as optim
import torch.nn.functional as F
from utils.siamese.util import *
class BYOLFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Dict,moving_average_decay,learning_rate,ckpt):
        super().__init__(observation_spaces,features_dim=1)
        self.model = BYOL(observation_spaces,moving_average_decay)
        self.optimizer =  optim.Adam(self.model.parameters(),lr=learning_rate)
        self._features_dim = self.model._features_dim
        self.scaler = None
        if(ckpt is not None):
            self.check_point = th.load(ckpt)
            self.load_state_dict(self.check_point['state_dict'])
    def forward(self,observations):
        return self.model(observations)
    def compute_loss(self,obs,obs_):
        return self.model.compute_loss(obs,obs_)



def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

class BYOL(nn.Module):
    def __init__(self, observation_spaces: spaces.Dict,moving_average_decay):
        super().__init__()
        extractors = {}
        projectors = {}
        predictors = {}
        total_concat_size = 0
        for key, subspace in observation_spaces.spaces.items():
            if key == "image" or key == "semantic" or key == "depth" or key == "gray" or key == "semanticlidar" or key == "infrared" :
                extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                #extractors[key] = torchvision.models.resnet50(zero_init_residual=True)
                with th.no_grad():
                    n_flatten = extractors[key](th.as_tensor(subspace.sample()[None]).float()).shape[1]
                    total_concat_size += n_flatten
                with th.no_grad():
                    p_flatten = extractors[key](th.ones((1, 3, 224, 224), dtype=th.float32)).shape[1]
                projectors[key] = nn.Sequential(
                    nn.Linear(p_flatten, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128,128),
                    nn.ReLU()
                )
                predictors[key] = nn.Sequential(
                    nn.Linear(128,256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256,128)
                )
            elif key == "ray" or key == "label" or key == "vec":
                extractors[key] = nn.Linear(subspace.shape[0], 256)
                projectors[key] = nn.Sequential(
                    nn.Linear(256,2048,bias=False),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048,2048,bias=False)
                )
                total_concat_size += 256
            elif key == "objectDet" or key == "speed" or key == "intention" or key == "state":
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        self.projectors = nn.ModuleDict(projectors)
        self.predictors = nn.ModuleDict(predictors)
        self.target_extractor = None
        self.target_ema_updater = EMA(moving_average_decay)
    def forward(self, observations):
        encoded_tensor_list = []
        for key, obs in observations.items():
            if key in set(it[0] for it in self.extractors.items()):
                encoded_tensor_list.append(self.extractors[key](observations[key]))
            else:
                encoded_tensor_list.append(observations[key])
        '''for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))'''
        return th.cat(encoded_tensor_list, dim=1)

    def compute_loss(self,observation,observation_):
        batch_size = 0
        losses = []
        loss = 0

        for key,obs in observation.items():
            if key in set(it[0] for it in self.projectors.items()):
                if batch_size == 0:
                    batch_size = obs.shape[0]
                input_1 = obs
                input_2 = observation_[key]
                z1 = self.extractors[key](input_1)
                #z2 = self.extractors[key](input_2)
                z1 = self.projectors[key](z1)
                predict_1 = self.predictors[key](z1)
                with th.no_grad():
                    # if self.target_extractor==None:
                    #     self.target_extractor = self.get_target_net()
                    self.target_extractor = self.extractors
                    z2 = self.target_extractor[key](input_2)
                    z2 = self.projectors[key](z2)
                losses.append(F.cosine_similarity(z2,predict_1))
        n = losses.__len__()

        for l in losses:
             loss += l*(1/n)
        update_moving_average(self.target_ema_updater,self.target_extractor,self.extractors)
        return loss.mean()
    @singleton('target_extractor')
    def get_target_net(self):
        target_extractor = copy.deepcopy(self.extractors)
        for p in target_extractor.parameters():
            p.requires_grad = False
        return target_extractor



class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)