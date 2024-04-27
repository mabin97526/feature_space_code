import math

import gymnasium
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gymnasium as gym
from gymnasium import spaces
import numpy as np
class UnityBaseEnv(gym.Env):
    def __init__(self,file_name:str,worker_id:int,time_scale:int=1,max_steps=2500):

        self.file_name = file_name
        self.worker_id = worker_id
        self.time_scale = time_scale
        self.channel = EngineConfigurationChannel()
        Unity_env = UnityEnvironment(file_name=file_name,
                                     worker_id=self.worker_id,
                                     side_channels=[self.channel])

        self._env = UnityToGymWrapper(Unity_env,uint8_visual=True,flatten_branched=False,allow_multiple_obs=True)
        self.channel.set_configuration_parameters(
            width=800,
            height=800,
            quality_level=5,
            time_scale=self.time_scale
            )
        self.max_steps = max_steps

    def seed(self, seed):
        self._env.seed(seed)

    @property
    def observation_space(self):

        #print(self._env.observation_space)

        obs_1 = spaces.Box(low=0,high=255,shape=(84,84,3),dtype=np.uint8)
        obs_2 = spaces.Box(low=-math.inf,high=math.inf,shape=(808,),dtype=np.float32)

        #obs_shape = gymnasium.spaces.Tuple((obs_1,obs_2),seed=42)
        obs_shape = gymnasium.spaces.Dict(spaces={"image": obs_1,"ray":obs_2}, seed=42)
        #print(obs_shape)
        #self._env.observation_space
        #return self._env.observation_space
        return obs_shape

    @property
    def action_space(self):
        #self._env.action_space
        #print(self._env.action_space)
        #action_space = spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype=np.float32)
        action_space = spaces.MultiDiscrete(np.array([3,3]))
        return action_space

    def reset(self):
        self._rewards = []
        #obs2 = self._env.reset()
        obs = self._env.reset()
        obs = {"image": obs[0],"ray": obs[1]}
        return obs,{}

    def step(self,action):

        #action[0] = 0.8
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        if done:
            info = {'reward': sum(self._rewards),
                    'length': len(self._rewards)}
        else:
            info = {}
        if(len(self._rewards)==self.max_steps):
            truncated = True
        else:
            truncated = False
        reward = float(reward)
        obs = {"image":obs[0],"ray":obs[1]}
        return obs,reward,done,truncated,info

    def close(self):
        self._env.close()

