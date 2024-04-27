from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union, List

import torch as th
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.redq.policies import Actor,CnnPolicy,MlpPolicy,MultiInputPolicy,REDQPolicy
from gymnasium import spaces
SelfREDQ = TypeVar("SelfREDQ", bound="REDQ")

class REDQ(OffPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: REDQPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    def __init__(
            self,
            policy: Union[str, Type[REDQPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            utd_ratio: int = 20,
            num_min: int = 2,
            q_target_mode: str = 'min',
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            policy_update_delay: int = 20,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None
        self.q_target_mode = q_target_mode
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.utd_ratio = utd_ratio
        self.num_min = num_min
        self._n_updates = 0  # type: int
        self.policy_update_delay = policy_update_delay

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
    def get_redq_q_target_no_grad(self, next_obs, rewards, dones):
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.policy.critic_kwargs["n_critics"],num_mins_to_use,replace=False)
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            ent_coef = th.exp(self.log_ent_coef.detach())
        else:
            ent_coef = self.ent_coef_tensor
        with th.no_grad():
            if self.q_target_mode == 'min':
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.reshape(-1, 1)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.critic_target.qn_forward(next_obs,next_actions,sample_idx)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = th.cat(q_prediction_next_list,1)
                min_q,min_indices = th.min(q_prediction_next_cat,dim=1,keepdim=True)
                next_q_with_log_prob = min_q - ent_coef * next_log_prob
                y_q = rewards + self.gamma * (1 - dones) * next_q_with_log_prob
            if self.q_target_mode == 'ave':
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.reshape(-1, 1)
                q_prediction_next_list = []
                for idx in range(self.policy.critic_kwargs["n_critics"]):
                    q_prediction_next = self.critic_target.qn_forward(next_obs,next_actions,idx)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_ave = th.cat(q_prediction_next_list,1).mean(dim=1).reshape(-1,1)
                next_q_with_log_prob = q_prediction_next_ave - ent_coef * next_log_prob
                y_q = rewards + self.gamma * (1-dones) * next_q_with_log_prob
            if self.q_target_mode == 'rem':
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.reshape(-1, 1)
                q_prediction_next_list = []
                for idx in range(self.policy.critic_kwargs["n_critics"]):
                    q_prediction_next = self.critic_target.qn_forward(next_obs,next_actions,idx)
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = th.cat(q_prediction_next_list, 1)
                rem_weight = th.Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.device)
                normalize_sum = rem_weight.sum(1).reshape(-1,1).expand(-1,self.policy.critic_kwargs["n_critics"])
                rem_weight /= normalize_sum
                q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1,1)
                next_q_with_log_prob = q_prediction_next_rem - ent_coef * next_log_prob
                y_q = rewards + self.gamma * (1 - dones) * next_q_with_log_prob
        return y_q, sample_idxs





    def train(self,gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            if self.use_sde:
                self.actor.reset_noise()

            ent_coef_loss = None
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            y_q, sample_idxs = self.get_redq_q_target_no_grad(replay_data.next_observations,replay_data.rewards,replay_data.dones)
            q_prediction_list = []
            for q_i in range(self.policy.critic_kwargs["n_critics"]):
                q_prediction = self.critic.qn_forward(replay_data.observations,replay_data.actions,q_i)
                q_prediction_list.append(q_prediction)
            q_prediction_cat = th.cat(q_prediction_list,dim=1)
            y_q = y_q.expand((-1, self.policy.critic_kwargs["n_critics"])) if y_q.shape[1] == 1 else y_q
            q_loss_all = F.mse_loss(q_prediction_cat,y_q) * self.policy.critic_kwargs["n_critics"]
            critic_losses.append(q_loss_all.item())
            self.critic.optimizer.zero_grad()
            q_loss_all.backward()
            if (self._n_updates + 1) % self.policy_update_delay == 0:
                q_list = []
                for sample_idx in range(self.policy.critic_kwargs["n_critics"]):
                    self.critic.q_networks[sample_idx].requires_grad_(False)
                    q_v = self.critic.qn_forward(replay_data.observations,actions_pi,sample_idx)
                    q_list.append(q_v)
                q_list_cat = th.cat(q_list,1)
                ave_q = th.mean(q_list_cat,dim=1,keepdim=True)
                actor_loss = (ent_coef * log_prob - ave_q).mean()
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.logger.record("train/actor_loss", np.mean(actor_losses))
                for sample_idx in range(self.policy.critic_kwargs["n_critics"]):
                    self.critic.q_networks[sample_idx].requires_grad_(True)

            self.critic.optimizer.step()

            if gradient_step % self.target_update_interval == 0 :
                polyak_update(self.critic.parameters(),self.critic_target.parameters(),self.tau)
                polyak_update(self.batch_norm_stats,self.batch_norm_stats_target,1.0)

        self._n_updates += gradient_steps


        self.logger.record("train/n_updates",self._n_updates,exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfREDQ,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "REDQ",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfREDQ:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins