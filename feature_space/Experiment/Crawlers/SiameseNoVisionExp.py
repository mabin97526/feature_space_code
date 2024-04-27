import time
import UnityEnvClass.Unity_PushBlock_env,UnityEnvClass.Unity_CrawlerNoVision_env
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from utils.customCallback import TensorboardCallback
from utils.custom_extractors import CustomCombinedExtractor
from utils.siamese.BarlowTwins import BarlowTwinsFeatureExtractor

import UnityEnvClass.Unity_Crawler_env
if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env_name = "CrawlersNoV"
    Algorithm = "PPO"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm, env_name) + timestamp,
                           ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_CrawlerNoVision_env.UnityBaseEnv, n_envs=5,start_index=15, env_kwargs=dict(file_name="../../UnityEnv/CrawlerNoVision/UnityEnvironment.exe", time_scale=10,max_steps=1000))
    # policy_kwargs = dict(features_extractor_class=BarlowTwinsFeatureExtractor,
    #                      features_extractor_kwargs=dict(lambd=0.0051))
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(use_resnet=False))
    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda", learning_rate=3e-4, seed=42, gamma=0.99,
                policy_kwargs=policy_kwargs, use_siamese=False)
    model.set_logger(new_logger)
    model.learn(total_timesteps=30000000, log_interval=4, callback=TensorboardCallback())