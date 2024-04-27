import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from utils.Intention.ActionPredictNetwork import *
from utils.Intention.BehaviourNetwork import *
SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        intention_enable: bool = False,
        save_opitimizer:bool = True,
        save_reward:float = 9.0,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.intention_enable = intention_enable
        self.save_optimizer = save_opitimizer
        self.save_reward = save_reward
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            intention_enable=self.intention_enable
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)
        if self.intention_enable:
            self.ActionEncoder = RNNEncoder(2,128,64,1).to(self.device)
            self.ActionPredictor = Action_Latent_Decoder(64,128,1,2,2).to(self.device)
            self.action_hidden_1 = th.zeros((1,128),dtype=th.float32).to(self.device)
            self.action_hidden_2 = th.zeros((1,128),dtype=th.float32).to(self.device)
            self.action_prev_h_1 = th.zeros((1,64),dtype=th.float32).to(self.device)
            self.action_prev_h_2 = th.zeros((1,64),dtype=th.float32).to(self.device)
            self.decoder_hidden_1 = th.zeros((1,128),dtype=th.float32).to(self.device)
            self.decoder_hidden_2 = th.zeros((1,128),dtype=th.float32).to(self.device)


            self.BehaviourPredictor = BehaviourNet(2,64,2,64).to(self.device)
            self.prev_hidden_state_1 = th.zeros((1,16,64),dtype=th.float32).to(self.device)
            self.hidden_state_1 = th.zeros((1,64),dtype=th.float32).to(self.device)
            self.prev_hidden_state_2 = th.zeros((1, 16, 64),dtype=th.float32).to(self.device)
            self.hidden_state_2 = th.zeros((1, 64),dtype=th.float32).to(self.device)
            # 优化器
            # self.intention_params = list(self.ActionEncoder.parameters())\
            #                          +list(self.ActionPredictor.parameters())\
            #                          +list(self.BehaviourPredictor.parameters())
            #self.intention_params = list(self.BehaviourPredictor.parameters())
            self.intention_params = list(self.ActionEncoder.parameters()) \
                                       +list(self.ActionPredictor.parameters())
            self.intention_optimizer = th.optim.Adam(self.intention_params,
                                                     lr=3e-4,eps=0.00001
                                                     )
            self.behaviour_intention_acc = 0
            self.action_intention_acc = 0
            self.obs_predic = []
            self.obs_ = []

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        intention_enable: bool = False
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            b_in = None
            a_in = None
            e_s = None
            e_a = None
            p_s = None
            p_a = None
            p_h_s = None
            h_s = None
            a_h = None
            d_h = None
            a_p_h = None
            if intention_enable:
                with th.no_grad():
                    enemy1_action = new_obs["state"][0][0:2]
                    enemy1_state = new_obs["state"][0][2:4]
                    enemy2_action = new_obs["state"][0][4:6]
                    enemy2_state = new_obs["state"][0][6:8]
                    e_s1 = th.as_tensor(enemy1_state,dtype=th.float32,device=self.device)
                    e_s2 = th.as_tensor(enemy2_state,dtype=th.float32,device=self.device)
                    e_a1 = th.as_tensor(enemy1_action,dtype=th.int8,device=self.device)
                    e_a2 = th.as_tensor(enemy2_action,dtype=th.int8,device=self.device)

                    enemy_act = np.concatenate((enemy1_action,enemy2_action),axis=0).reshape(-1,4)
                    enemy_stat = np.concatenate((enemy1_state,enemy2_state),axis=0).reshape(-1,4)

                    p_h_s = th.concatenate((self.prev_hidden_state_1,self.prev_hidden_state_2),dim=1).cpu().numpy() # 1* 32 *64
                    h_s = th.concatenate((self.hidden_state_1,self.hidden_state_2),dim=1).cpu().numpy() #1* 128
                    a_h = th.concatenate((self.action_hidden_1, self.action_hidden_2), dim=1).cpu().numpy()
                    d_h = th.concatenate((self.decoder_hidden_1, self.decoder_hidden_2), dim=1).cpu().numpy()
                    a_p_h = th.concatenate((self.action_prev_h_1, self.action_prev_h_2), dim=1).cpu().numpy()

                    # 行为意图计算模块
                    out1,new_hidden_state_1=self.BehaviourPredictor(e_s1,self.prev_hidden_state_1,self.hidden_state_1)
                    predicted_be1 = th.argmax(out1,dim=1)
                    predicted_be1 = F.one_hot(predicted_be1,num_classes=2)
                    self.hidden_state_1 = new_hidden_state_1
                    self.prev_hidden_state_1 = th.concatenate((self.prev_hidden_state_1,self.hidden_state_1.unsqueeze(0)),dim=1)
                    self.prev_hidden_state_1 = self.prev_hidden_state_1[0][self.prev_hidden_state_1.shape[1]-16:self.prev_hidden_state_1.shape[1]].unsqueeze(0)

                    out2, new_hidden_state_2 = self.BehaviourPredictor(e_s2, self.prev_hidden_state_2, self.hidden_state_2)
                    predicted_be2 = th.argmax(out2, dim=1)
                    predicted_be2 = F.one_hot(predicted_be2, num_classes=2)
                    self.hidden_state_2 = new_hidden_state_2
                    self.prev_hidden_state_2 = th.concatenate((self.prev_hidden_state_2, self.hidden_state_2.unsqueeze(0)),
                                                             dim=1)
                    self.prev_hidden_state_2 = self.prev_hidden_state_2[0][
                                              self.prev_hidden_state_2.shape[1] - 16:self.prev_hidden_state_2.shape[
                                                  1]].unsqueeze(0)

                    #取消行为意图将下面两行取消注释
                    #predicted_be1 = th.tensor([[0,0]],dtype=th.float32).to(self.device)
                    #predicted_be2 = th.tensor([[0,0]],dtype=th.float32).to(self.device)
                    encoder_input_1 = th.concatenate((e_s1.unsqueeze(0),predicted_be1),dim=1)
                    _,new_hidden_1,output1 = self.ActionEncoder(encoder_input_1,self.action_hidden_1)
                    self.action_hidden_1 = new_hidden_1

                    encoder_input_2 = th.concatenate((e_s2.unsqueeze(0),predicted_be2),dim=1)
                    _,new_hidden_2,output2 = self.ActionEncoder(encoder_input_2,self.action_hidden_2)
                    self.action_hidden_2 = new_hidden_2

                    o1, a11, a12, decoder_hidden_1 = self.ActionPredictor(output1, self.action_prev_h_1, self.decoder_hidden_1)
                    o2, a21, a22, decoder_hidden_2 = self.ActionPredictor(output2,self.action_prev_h_2,self.decoder_hidden_2)

                    a11 = th.argmax(a11.squeeze(0), dim=1)
                    a12 = th.argmax(a12.squeeze(0), dim=1)

                    a21 = th.argmax(a21.squeeze(0),dim=1)
                    a22 = th.argmax(a22.squeeze(0),dim=1)

                    self.decoder_hidden_1 = decoder_hidden_1.squeeze(0)
                    self.decoder_hidden_2 = decoder_hidden_2.squeeze(0)
                    self.action_prev_h_1 = output1
                    self.action_prev_h_2 = output2

                    b_in = th.concatenate((predicted_be1,predicted_be2),dim=1).cpu().numpy()
                    a_in = th.concatenate((output1,output2),dim=1).cpu().numpy()
                    e_s = np.concatenate((enemy1_state,enemy2_state))
                    e_a = np.concatenate((enemy1_action,enemy2_action))
                    p_s = th.concatenate((o1.squeeze(0),o2.squeeze(0)),dim=1).cpu().numpy()
                    p_a = th.concatenate((a11,a12,a21,a22),dim=0).unsqueeze(0).cpu().numpy()

                    if np.array_equal(b_in,np.array((0,1,1,0))):
                        self.behaviour_intention_acc+=1
                    if np.array_equal(p_a,enemy_act):
                        self.action_intention_acc += 1
                    self.obs_predic.append(p_s)
                    self.obs_.append(enemy_stat)
                    new_obs["intention"] = np.concatenate((b_in,a_in),axis=1)
                    #new_obs["intention"] = b_in

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):

                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value


            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                intention_enable ,
                b_in,
                a_in,
                e_s,
                e_a,
                p_s,
                p_a,
                a_h,
                d_h,
                a_p_h,
                p_h_s=p_h_s,
                h_s = h_s
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps,intention_enable=self.intention_enable)

            if continue_training is False:
                break
            #print(iteration)
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            if iteration % 50 == 0 and iteration != 0  :
                timestamp = time.strftime("/%Y%m%d-%H%M%S")
                self.save("results/"+timestamp +"ppo"+str(iteration) )
            if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) >= self.save_reward:
                self.save("results/"+"best"+"ppo"+str(iteration))
                self.save_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        if self.intention_enable:
            state_dicts = ["policy","policy.optimizer","ActionEncoder","ActionPredictor","BehaviourPredictor","intention_optimizer"]
        elif self.save_optimizer:
            state_dicts = ["policy", "policy.optimizer"]
        else:
            state_dicts = ["policy"]

        return state_dicts, []
