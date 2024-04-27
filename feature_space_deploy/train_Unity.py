import time
import torch
import torch.nn as nn

import UnityEnvClass.Unity_Multi_input_CarTraffic_Env
from UnityEnvClass import Unity_RollerBall, Unity_3Dball, Unity_3DballVision, Unity_Base_env
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *



if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    #env = UnityBaseEnv("UnityEnv/CarRace",worker_id=1,time_scale=1,max_steps=2500)
    #env = Unity_RollerBall.UnityBaseEnv("UnityEnv/rollerball",worker_id=1,time_scale=1,max_steps=1500)
    env_name = "CarTrafficPhaseI"
    #env = UnityEnvClass.Unity_Multi_input_CarTraffic_Env.UnityBaseEnv(file_name="./UnityEnv/CarTrafficPhase1",worker_id=1,time_scale=1,max_steps=2500)
    #env = Unity_Base_env.UnityBaseEnv(file_name="./UnityEnv/CarTrafficPhase1",worker_id=1,time_scale=1,max_steps=2500)
    new_logger = configure("./loggers/PPO_TrafficPhaseI_concat_{}_logger".format(env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    '''policy_kwargs = dict(features_extractor_class = CustomCnn,
                         features_extractor_kwargs=dict(features_dim=256),
                         net_arch=[128,128],
                         )'''
    env = make_unity_env(UnityEnvClass.Unity_Base_env.UnityBaseEnv, n_envs=2, env_kwargs=dict(file_name="UnityEnv/CarTrafficPhase1", time_scale=10,max_steps=2500))
    #policy_kwargs = dict(features_extractor_class = ResNetImageExtractor,features_extractor_kwargs = dict(output_shape=256),
    #                     net_arch=[128,128]
    #                     )
    #policy_kwargs = dict(features_extractor_class= LowRankTensorFusionExtractor,features_extractor_kwargs=dict(output_dim = 128,rank=10),share_features_extractor=False)
    #policy_kwargs = dict(share_features_extractor = False)
    #model = SAC("MultiInputPolicy", env, learning_rate=3e-4, buffer_size=300000, batch_size=512,
                #learning_starts=3000, verbose=1, seed=42)
    #policy_kwargs = dict(features_extractor_class = ResNetImageExtractor,share_features_extractor = False,features_extractor_kwargs=dict(output_shape=256))
    policy_kwargs = dict(share_features_extractor=False)
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
    #model = TD3("CnnPolicy",env,verbose=1,device="cuda",learning_rate=3e-4,action_noise=None,buffer_size=50000,learning_starts=5000,train_freq=(2,"episode"))
    model = PPO("CnnPolicy",env,verbose=1,device="cuda",learning_rate=3e-4,seed=42,gamma=0.99,policy_kwargs=policy_kwargs)
    #model = SAC("MlpPolicy",env,learning_rate=3e-4,buffer_size=10000,batch_size=3000,train_freq=(2,"episode"),learning_starts=10000,verbose=1,seed=42,policy_kwargs=policy_kwargs)
    #model = SAC("CnnPolicy", env, learning_rate=3e-4,buffer_size=20000, batch_size=3000,train_freq=(2, "episode"),learning_starts=6000, verbose=1,seed=42,policy_kwargs=policy_kwargs)
    model.set_logger(new_logger)
    model.learn(total_timesteps=10000000, log_interval=4,callback=TensorboardCallback())

    '''obs = env.reset()[0]
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # print(reward)
        #env.render()
        if done:
            obs = env.reset()[0]'''