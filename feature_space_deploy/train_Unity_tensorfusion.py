import time
import torch
import torch.nn as nn

import UnityEnvClass.Unity_Multi_input_CarRace_Env
from UnityEnvClass import Unity_RollerBall, Unity_3Dball, Unity_3DballVision
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from UnityEnvClass.Unity_RollerBall import *
from stable_baselines3.sac.custom_sac import Custom_SAC
from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *



if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    #env = UnityBaseEnv("UnityEnv/CarRace",worker_id=1,time_scale=1,max_steps=2500)
    #env = Unity_RollerBall.UnityBaseEnv("UnityEnv/rollerball",worker_id=1,time_scale=1,max_steps=1500)
    env_name = "CarRace"
    #env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarRace_Env.UnityBaseEnv, n_envs=1, env_kwargs=dict(file_name="UnityEnv/CarRace", time_scale=1,max_steps=2500))
    env = UnityEnvClass.Unity_Multi_input_CarRace_Env.UnityBaseEnv("UnityEnv/CarRace",worker_id=2)
    new_logger = configure("./loggers/TD3_concat_{}_logger".format(env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    '''policy_kwargs = dict(features_extractor_class = CustomCnn,
                         features_extractor_kwargs=dict(features_dim=256),
                         net_arch=[128,128],
                         )'''


    class HyperParams(MULTModel.DefaultHyperParams):
        num_heads = 2
        embed_dim = 256
        output_dim = 64
        all_steps = True
    #policy_kwargs = dict(features_extractor_class = TranformerFusionExtractor,share_features_extractor=False,features_extractor_kwargs=dict(hyper_param=HyperParams,ckpt="./utils/checkpoints/extractor_checkpoint_transformer_2900.pth.tar"))

    #policy_kwargs = dict(features_extractor_class = CustomCombinedExtractor,features_extractor_kwargs = dict(output_shape=256),
    #                     net_arch=[128,128]
    #                    )
    policy_kwargs = dict(features_extractor_class= CustomCombinedExtractor,features_extractor_kwargs=dict(use_resnet=True))
    #model = SAC("MultiInputPolicy", env, learning_rate=3e-4, buffer_size=300000, batch_size=512,
                #learning_starts=3000, verbose=1, seed=42)
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
    model = TD3("MultiInputPolicy",env,learning_rate= 1e-3,action_noise=action_noise,policy_kwargs=policy_kwargs,verbose=1,device="cuda",learning_starts=10000,buffer_size=30000,seed=42)

    #model = SAC("MlpPolicy",env,learning_rate=3e-4,buffer_size=10000,batch_size=3000,train_freq=(2,"episode"),learning_starts=10000,verbose=1,seed=42,policy_kwargs=policy_kwargs)
    #model = SAC("CnnPolicy", env, learning_rate=3e-4,buffer_size=10000, batch_size=3000,train_freq=(2, "episode"),learning_starts=6000, verbose=1,seed=42)
    model.set_logger(new_logger)
    model.learn(total_timesteps=5000000, log_interval=4,callback=TensorboardCallback())
    model.save(env_name+timestamp)
    '''obs = env.reset()[0]
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # print(reward)
        #env.render()
        if done:
            obs = env.reset()[0]'''