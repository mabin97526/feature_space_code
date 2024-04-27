import torch.nn as nn

import UnityEnvClass.Unity_Multi_input_CarRace_NormalEnv,UnityEnvClass.Unity_Multi_input_CarRace_LidarEnv,UnityEnvClass.Unity_PushBlock_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from UnityEnvClass.Unity_Base_env import *
from UnityEnvClass.Unity_PushBlock_env import *
import UnityEnvClass.Unity_Crawler_env
import UnityEnvClass.Unity_WallJump_env
import UnityEnvClass.Unity_Multi_input_CarSearch_Detection_Random_Env
import UnityEnvClass.Unity_Multi_input_CarSearch_InfraredEnv
import cv2
steps = 0
class TensorboardCallback(BaseCallback):
    def __init__(self,verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        pass
        #print(self.locals["rewards"])
        #print(steps)

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self,observation_space:spaces.Box,features_dim):
        super().__init__(observation_space,features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn=nn.Sequential(

        )
if __name__ == '__main__':
    '''timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env = make_vec_env("HalfCheetah",n_envs=8)
    #env = gym.make("CarRacing-v2",render_mode="human")
    new_logger = configure("./loggers/SAC_Ant_logger"+timestamp,["stdout","csv","tensorboard"])

    model = SAC("MlpPolicy",env,learning_starts=1000,verbose=1,gamma = 0.9,batch_size=256,buffer_size=100000,train_freq=4,tau=0.001,gradient_steps=4,ent_coef="auto",target_entropy="auto",use_sde=True)

    model.set_logger(new_logger)
    model.learn(total_timesteps=1000000,log_interval=1)
    model.save("sac_ant")
    obs = env.reset()
    while True:
        action,_states = model.predict(obs,deterministic=True)
        obs,reward,done,truncated,info = env.step(action)
        #print(reward)
        env.render()
        if done:
            obs,info = env.reset()'''
    env = UnityEnvClass.Unity_Crawler_env.UnityBaseEnv(file_name="./UnityEnv/Crawler", worker_id=16)
    shape = env.observation_space
    obs = env.reset()
    done = False
    action = env.action_space.sample()
    e_r = 0
    while True:
        obs,reward,done,truncated,info = env.step(action)
        cv2.imshow("obs1", obs['image'])
        cv2.waitKey(1)
        e_r += reward
        action = env.action_space.sample()
        env.reset()
        if done:
            env.reset()
            done = False
            print(e_r)
            e_r=0
    env.close()

    '''
     0 rgb 1 semantic
    cv2.imshow("obs1",obs['semantic'])
    cv2.waitKey(0)
    print(env.action_space)
    print(env.observation_space)
    env.close()
    '''
