import math

import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import *
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.optim as optim
import torch.nn.functional as F
from utils.siamese.util import *
class VICRegAPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_spaces:spaces.Dict,base_lr:float,weight_decay:float,sim_coeff:float,std_coeff,cov_coeff,ckpt,action_space):
        super().__init__(observation_spaces,features_dim=1)
        self.model = VICRegAP(observation_spaces, sim_coeff, std_coeff, cov_coeff, action_space)
        self.optimizer = LARS(self.model.parameters(),lr=0,weight_decay=weight_decay,
                         weight_decay_filter=exclude_bias_and_norm,
                         lars_adaptation_filter=exclude_bias_and_norm
                         )
        self._features_dim = self.model._features_dim
        self.scaler = th.cuda.amp.GradScaler()
        self.base_lr = base_lr
        if(ckpt is not None):
            self.check_point = th.load(ckpt)
            self.load_state_dict(self.check_point['state_dict'])
    def forward(self,observations):
        return self.model(observations)
    def compute_loss(self,obs,obs_):
        return self.model.compute_loss(obs,obs_)
    def adjust_learning_rate(self,optimizer,step,max_step,b_lr):
        return adjust_learning_rate(optimizer,step,max_step,b_lr)
class VICRegAP(nn.Module):
    def __init__(self, observation_spaces: spaces.Dict,sim_coeff:float,std_coeff,cov_coeff,action_space):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        extractors = {}
        projectors = {}
        total_concat_size = 0
        self.p_flatten_size = 0
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
                if key == "image" or key == "semantic" or key == "depth":
                    with th.no_grad():
                        p_flatten = extractors[key](th.ones((1, subspace.shape[0], 224, 224), dtype=th.float32)).shape[1]
                        self.p_flatten_size+=p_flatten
                    projectors[key] = nn.Sequential(
                        nn.Linear(p_flatten, 2048),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Linear(2048, 2048, bias=False)
                    )
                else:
                    self.p_flatten_size+=n_flatten
            elif key == "ray" or key == "label" or key == "vec":
                extractors[key] = nn.Linear(subspace.shape[0], 256)
                projectors[key] = nn.Sequential(
                    nn.Linear(256,2048,bias=False),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048,2048,bias=False)
                )
                total_concat_size += 256
                self.p_flatten_size+=256
            elif key == "objectDet" or key == "speed" or key == "intention" or key == "state":
                total_concat_size += subspace.shape[0]
                self.p_flatten_size+=subspace.shape[0]


        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        self.projectors = nn.ModuleDict(projectors)
        self.acion_predictor, self.a_std = None,None
        if action_space is not None:
            self.build_act(action_space,use_sde=False,dist_kwargs=None)


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
        encoded_tensor_list = []
        encoded_tensor_aug_list = []
        for key,obs in observation.items():
            if key in set(it[0] for it in self.projectors.items()):
                if batch_size == 0:
                    batch_size = obs.shape[0]
                input_1 = obs
                input_2 = observation_[key]
                z1 = self.extractors[key](input_1)
                encoded_tensor_list.append(z1)
                z2 = self.extractors[key](input_2)
                encoded_tensor_aug_list.append(z2)
                z1 = self.projectors[key](z1)
                z2 = self.projectors[key](z2)
                # empirical cross-correlation matrix
                repr_loss = F.mse_loss(z1,z2)

                x = z1 - z1.mean(dim=0)
                y = z2 - z2.mean(dim=0)

                std_x = th.sqrt(x.var(dim=0) + 0.0001)
                std_y = th.sqrt(y.var(dim=0) + 0.0001)
                std_loss = th.mean(F.relu(1-std_x)) / 2 + th.mean(F.relu(1-std_y)) / 2

                cov_x = (x.T @ x) / (batch_size - 1)
                cov_y = (y.T @ y) / (batch_size - 1)
                cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(2048) + self.off_diagonal(cov_y).pow_(2).sum().div(2048)

                losses.append(self.sim_coeff * repr_loss
                              + self.std_coeff * std_loss
                              + self.cov_coeff * cov_loss)
            elif key in set(it[0] for it in self.extractors.items()):
                encoded_tensor_list.append(self.extractors[key](observation[key]))
                encoded_tensor_aug_list.append(self.extractors[key](observation[key]))
            else:
                encoded_tensor_list.append(observation[key])
                encoded_tensor_aug_list.append(observation[key])

        n = losses.__len__()
        action_loss = 0
        if self.acion_predictor != None:
            mean_action = self.acion_predictor(th.cat(encoded_tensor_list, dim=1))
            mean_action_aug = self.acion_predictor(th.cat(encoded_tensor_aug_list,dim=1))
            action_loss = F.cosine_similarity(mean_action,mean_action_aug)
        for l in losses:
             loss += l*(1/n)
        loss += action_loss.mean()
        return loss

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def make_proba_distribution(self,
            action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
    ) -> Distribution:
        """
        Return an instance of Distribution for the correct type of action space

        :param action_space: the input action space
        :param use_sde: Force the use of StateDependentNoiseDistribution
            instead of DiagGaussianDistribution
        :param dist_kwargs: Keyword arguments to pass to the probability distribution
        :return: the appropriate Distribution object
        """
        if dist_kwargs is None:
            dist_kwargs = {}

        if isinstance(action_space, spaces.Box):
            cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
            return cls(get_action_dim(action_space), **dist_kwargs)
        elif isinstance(action_space, spaces.Discrete):
            return CategoricalDistribution(action_space.n, **dist_kwargs)
        elif isinstance(action_space, spaces.MultiDiscrete):
            return MultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)
        elif isinstance(action_space, spaces.MultiBinary):
            return BernoulliDistribution(action_space.n, **dist_kwargs)
        else:
            raise NotImplementedError(
                "Error: probability distribution, not implemented for action space"
                f"of type {type(action_space)}."
                " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
            )

    def build_act(self,action_space,use_sde,dist_kwargs ):
        distribution = self.make_proba_distribution(action_space,use_sde,dist_kwargs)
        if isinstance(distribution, DiagGaussianDistribution):
            self.acion_predictor, self.a_std = distribution.proba_distribution_net(
                latent_dim=self.p_flatten_size, log_std_init=0.0
            )
        elif isinstance(distribution, StateDependentNoiseDistribution):
            self.acion_predictor, self.a_std = distribution.proba_distribution_net(
                latent_dim=self.p_flatten_size, latent_sde_dim=self.p_flatten_size, log_std_init=0.0
            )
        elif isinstance(distribution,
                        (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.acion_predictor = distribution.proba_distribution_net(latent_dim=self.p_flatten_size)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

class LARS(optim.Optimizer):
    def __init__(self,
                 params,
                 lr,
                 weight_decay=0,
                 momentum=0.9,
                 eta=0.001,
                 weight_decay_filter=None,
                 lars_adaptation_filter=None
                 ):
        defaults = dict(lr=lr,weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params,defaults)

    @th.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = th.norm(p)
                    update_norm = th.norm(dp)
                    one = th.ones_like(param_norm)
                    q = th.where(param_norm > 0.,
                                    th.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = th.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def exclude_bias_and_norm(p):
    return p.ndim == 1

