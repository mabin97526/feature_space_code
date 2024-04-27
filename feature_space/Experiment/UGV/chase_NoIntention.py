import time


import UnityEnvClass.Unity_chase_intention
import UnityEnvClass.Unity_chase_semantic
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.tqc.tqc import TQC
from utils.custom_extractors import *

from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *

if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env_name = "CarChaseVisualSemantic"
    Algorithm = "PPO"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm, env_name) + timestamp,
                           ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_chase_semantic.UnityBaseEnv, n_envs=5,
                         env_kwargs=dict(file_name="../../UnityEnv/CarChase_Semantic/605.exe", time_scale=10,
                                         max_steps=2500))
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(use_resnet=False))
    agent = PPO("MultiInputPolicy",env,n_steps=512,verbose=1,device="cuda",learning_rate=3e-4,seed=7663,gamma=0.99,policy_kwargs=policy_kwargs,intention_enable=False,batch_size=32)
    agent.set_logger(new_logger)
    agent.learn(total_timesteps=5000000, log_interval=4,callback=TensorboardCallback())