import gymnasium as gym
import numpy as np
import time

class CarRacing:
    def __init__(self,env_name,render=False):
        render_mode = "human" if render else "rgb_array"
        self._env = gym.make(env_name,render_mode=render_mode)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards=[]
        obs,_ = self._env.reset()
        return obs

    def step(self,action):

        obs,reward,done,truncation,info = self._env.step(action)
        self._rewards.append(reward)
        if done or truncation:
            info = {"reward":sum(self._rewards),
                    "length":len(self._rewards)}
        else:
            info = None

        return obs,reward,done or truncation,info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()

