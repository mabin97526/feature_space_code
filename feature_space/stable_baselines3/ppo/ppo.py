import warnings
from datetime import time
from typing import Any, Dict, Optional, Type, TypeVar, Union
from torchvision import transforms as T
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from utils.siamese.Transform import Transform, GaussianNoise, SaltAndPepperNoise, DepthNoise, DepthSaltAndPepperNoise, \
    GaussianBlur, Solarization
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from PIL import Image
SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        intention_enable: bool = False,
        use_siamese: bool = False,
        adjust_learning_rate=False,
        save_optimizer=True,
        save_reward = 9.0
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            intention_enable=intention_enable,
            save_opitimizer=save_optimizer,
            save_reward = save_reward
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.use_siamese = use_siamese
        self.vis_cam_random_transformers = None
        self.adjust_learning_rate = adjust_learning_rate
        self.save_optimizer = save_optimizer
        if use_siamese:
            self.transform = T.Compose([
                #T.ToPILImage(),
                T.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                T.RandomGrayscale(p=0.2),
                #GaussianBlur(p=1.0),
                #Solarization(p=0.0),
                #T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform_prime = T.Compose([
                #T.ToPILImage(),
                T.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                T.RandomGrayscale(p=0.2),
                #GaussianBlur(p=0.1),
                #Solarization(p=0.2),
                #T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        behaviour_intention_loss = []
        action_intention_loss = []

        siamese_loss = []
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            if self.use_siamese:
                self.policy.features_extractor.requires_grad_(False)
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()


            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        if self.use_siamese :
            self.policy.features_extractor.requires_grad_(True)
            if self.adjust_learning_rate:
                self.policy.features_extractor.adjust_learning_rate(self.policy.features_extractor.optimizer,self.num_timesteps,5000000,self.policy.features_extractor.base_lr)
            observation = rollout_data.observations
            observation_ = {}
            del_keys = []
            for key, obs in observation.items():
                if key == "semantic":
                    rand_key = key + "_random"
                    observation_[key] = observation[rand_key]
                    del_keys.append(rand_key)
                elif key == "semantic_random":
                    continue
                if key == "image" :
                     aut_image = self.transform_prime(obs)
                     observation[key] = self.transform(obs)
                     observation_[key] = aut_image
                elif key == "ray":
                    ray = obs
                    ray_random = (th.rand(1) * 802).int()
                    random_indices = th.randperm(802, dtype=th.int64, device=self.device)[:ray_random]
                    ray[..., random_indices] = 1.
                    observation_[key] = ray
            for key in del_keys:
                del observation[key]
            # print(observation)
            self.policy.features_extractor.zero_grad()
            with th.cuda.amp.autocast():
                siam_loss = self.policy.features_extractor.compute_loss(observation, observation_)
            siamese_loss.append(siam_loss.item())
            if self.policy.features_extractor.scaler is not None:
                self.policy.features_extractor.scaler.scale(siam_loss).backward()
                self.policy.features_extractor.scaler.step(self.policy.features_extractor.optimizer)
                self.policy.features_extractor.scaler.update()
            else:
                siam_loss.backward()
                self.policy.features_extractor.optimizer.step()
        if  self._n_updates%5==0 and self.intention_enable:
            for rollout_data in self.rollout_buffer.get(self.batch_size,intention_enable=True):
                n = rollout_data.behaviour_intention.shape[0]
                repeat_times = int(n/4)
                #predict_ = rollout_data.behaviour_intention
                enemy_state = rollout_data.enemy_state
                enemy_action = rollout_data.enemy_action
                action_hidden = rollout_data.action_hidden
                decoder_hidden = rollout_data.decoder_hidden
                action_prev_hidden = rollout_data.action_prev_hidden
                prev_hidden_state = rollout_data.prev_hidden_state
                hidden_state = rollout_data.hidden_state

                a_h_1 = action_hidden[:,0:128]
                a_h_2 = action_hidden[:,128:256]
                d_h_1 = decoder_hidden[:,0:128]
                d_h_2 = decoder_hidden[:,128:256]
                a_p_h_1 = action_prev_hidden[:,0:64]
                a_p_h_2 = action_prev_hidden[:,64:128]
                p_h_s_1 = prev_hidden_state[:,0:16]
                p_h_s_2 = prev_hidden_state[:,16:32]
                h_s_1 = hidden_state[:,0:64]
                h_s_2 = hidden_state[:,64:128]

                enemy_state_1 = enemy_state[:,0:2]
                enemy_state_2 = enemy_state[:,2:4]

                real_behaviour_intention = th.tensor([0,1,1,0],dtype=th.float32).repeat(repeat_times).to(self.device)

                
                out1, _ = self.BehaviourPredictor(enemy_state_1, p_h_s_1, h_s_1)
                out2, _ = self.BehaviourPredictor(enemy_state_2, p_h_s_2, h_s_2)
                predicted_be1 = th.argmax(out1, dim=1)
                predicted_be1 = F.one_hot(predicted_be1, num_classes=2)
                predicted_be2 = th.argmax(out1, dim=1)
                predicted_be2 = F.one_hot(predicted_be2, num_classes=2)
                predict_ = th.concatenate((out1,out2),dim=1).flatten()
                behaviour_loss = F.binary_cross_entropy(predict_,real_behaviour_intention).mean()
                behaviour_intention_loss.append(behaviour_loss.item())

                # fake intention
                #predicted_be1 = th.zeros((64,2),dtype=th.float32).to(self.device)
                #predicted_be2 = th.zeros((64,2),dtype=th.float32).to(self.device)


                encoder_input_1 = th.concatenate((enemy_state[:,0:2], predicted_be1), dim=1).unsqueeze(1)
                _, _, output1 = self.ActionEncoder(encoder_input_1, a_h_1.unsqueeze(0).contiguous())
                # output1: 1*64
                encoder_input_2 = th.concatenate((enemy_state[:,2:4], predicted_be2), dim=1).unsqueeze(1)
                _, _, output2 = self.ActionEncoder(encoder_input_2, a_h_2.unsqueeze(0).contiguous())

                o1, a11, a12, decoder_hidden_1 = self.ActionPredictor(output1, a_p_h_1,
                                                                      d_h_1)
                o2, a21, a22, decoder_hidden_2 = self.ActionPredictor(output2, a_p_h_2,
                                                                      d_h_2)
                a11=a11.squeeze(1)
                a12=a12.squeeze(1)
                a21=a21.squeeze(1)
                a22=a22.squeeze(1)
                a11 = th.argmax(a11.squeeze(1), dim=1)
                a12 = th.argmax(a12.squeeze(1), dim=1)

                a21 = th.argmax(a21.squeeze(1), dim=1)
                a22 = th.argmax(a22.squeeze(1), dim=1)

                predict_action = th.concatenate((a11,a12,a21,a22),dim=0)
                predict_state = th.concatenate((o1.squeeze(1),o2.squeeze(1)),dim=1)
                state_loss = F.mse_loss(predict_state,enemy_state).mean()
                action_loss = F.mse_loss(predict_action,enemy_action).mean()
                total_ = state_loss*0.2 + action_loss*0.8
                action_intention_loss.append(total_.item())
                total_loss = total_*0.5 + behaviour_loss*0.5
                #total_loss = behaviour_loss
                #total_loss = total_
                self.intention_optimizer.zero_grad()
                total_loss.backward()
                self.intention_optimizer.step()
            # current_memory = th.cuda.memory_allocated(device=self.device)
            # th.cuda.memory_snapshot()
        # Logs

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        if self.intention_enable:
            self.logger.record("train/behaviour_intention_loss",np.mean(behaviour_intention_loss))
            self.logger.record("train/action_intention_loss",np.mean(action_intention_loss))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.use_siamese:
            self.logger.record("train/siamese_loss",np.mean(siamese_loss))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


