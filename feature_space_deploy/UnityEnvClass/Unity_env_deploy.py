import math

import gymnasium
 

import gymnasium as gym
from gymnasium import spaces
import numpy as np
class UnityBaseEnv(gym.Env):
    def __init__(self,file_name:str,worker_id:int,time_scale:int=1,max_steps=2500):

        self.file_name = file_name
        self.worker_id = worker_id
        self.time_scale = time_scale

        self.max_steps = max_steps

    def seed(self, seed):
        pass

    @property
    def observation_space(self):

        #print(self._env.observation_space)

        obs_1 = spaces.Box(low=0,high=255,shape=(84,84,3),dtype=np.uint8)
        obs_2 = spaces.Box(low=-math.inf,high=math.inf,shape=(802,),dtype=np.float32)

        #obs_shape = gymnasium.spaces.Tuple((obs_1,obs_2),seed=42)
        obs_shape = gymnasium.spaces.Dict(spaces={"image": obs_1, "semantic":obs_1,"ray":obs_2}, seed=42)
        #print(obs_shape)
        #self._env.observation_space
        #return self._env.observation_space
        return obs_shape

    @property
    def action_space(self):
        #self._env.action_space
        #print(self._env.action_space)
        action_space = spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype=np.float32)
        return action_space

    def reset(self):
        pass

    def step(self,action):
        pass

    def close(self):
        pass

