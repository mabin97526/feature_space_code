from typing import Any, Dict,List,Optional,Tuple,Type,Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn as nn

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Critic(BaseModel):
    """
    Critic network (q-value function) for REDQ.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """
    action_space: spaces.Box
    features_extractor: BaseFeaturesExtractor

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: List[int],
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 10,
            share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.q_networks = []
        self.n_critics = n_critics

        for i in range(n_critics):
            qf_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            qf_net = nn.Sequential(*qf_net_list)
            self.add_module(f"qf{i}", qf_net)
            self.q_networks.append(qf_net)

    def forward(self, obs: PyTorchObs, action: th.Tensor) -> th.Tensor:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, action], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)




class Actor(BasePolicy):
    action_space: spaces.Box

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

class REDQPolicy(BasePolicy):
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Box,
                 lr_schedule: Schedule,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 n_critics: int = 10,
                 share_features_extractor: bool = False,
        ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        # self.critic = []
        # self.target_critics = []
        # self.critic_params = []
        # self.q_optimizers = []
        # self.num_min = num_min
        # self.q_target_mode = q_target_mode
        # self.policy_update_delay = policy_update_delay
        if net_arch is None:
            net_arch = [256,256]
        actor_arch , critic_arch = get_actor_critic_arch(net_arch)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            critic_parameters = [param for name, param in self.critic.named_parameters() if
                                 "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_target.set_training_mode(False)

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr = lr_schedule(1),
            **self.optimizer_kwargs,
        )


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

MlpPolicy = REDQPolicy

class CnnPolicy(REDQPolicy):


    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 10,
        share_features_extractor: bool = False,
        num_min: int = 2,
        q_target_mode: str = 'min',
        policy_update_delay: int = 20,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            num_min,
            q_target_mode,
            policy_update_delay,
        )


class MultiInputPolicy(REDQPolicy):


    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 10,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,

        )
