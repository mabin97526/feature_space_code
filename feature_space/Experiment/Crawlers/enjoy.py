import time

import cv2
import numpy as np

import UnityEnvClass.Unity_Crawler_env
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from utils.customCallback import TensorboardCallback
from utils.custom_extractors import CustomCombinedExtractor
from utils.siamese.BYOL import BYOLFeatureExtractor
from utils.siamese.BarlowTwins import BarlowTwinsFeatureExtractor
from utils.siamese.VICREGModel import VICRegFeatureExtractor
from utils.siamese.VICREG_AP_Model import VICRegAPFeatureExtractor

if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env_name = "CralwerSiamese"
    Algorithm = "PPO"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm, env_name) + timestamp,
                           ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_Crawler_env.UnityBaseEnv, n_envs=1,start_index=0, env_kwargs=dict(file_name=None, time_scale=1,max_steps=1000))
    # policy_kwargs = dict(features_extractor_class=BarlowTwinsFeatureExtractor,
    #                      features_extractor_kwargs=dict(lambd=0.0051))
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False))
    # policy_kwargs = dict(features_extractor_class = VICRegFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01,weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None)
    #                      )
    # policy_kwargs = dict(
    #     features_extractor_class=BYOLFeatureExtractor,
    #     features_extractor_kwargs=dict(moving_average_decay=0.99, learning_rate=3e-4, ckpt=None)
    # )
    # policy_kwargs = dict(features_extractor_class=VICRegAPFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01, weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None, action_space=env.action_space)
    #                      )
    model = PPO("MultiInputPolicy", env, verbose=1, device="cpu", learning_rate=3e-4, seed=42, gamma=0.99,
                policy_kwargs=policy_kwargs, use_siamese=True)

    model.set_parameters(load_path_or_dict="results/vanillaPPO.zip")
    times = 20
    episode_rewards = []
    episode_lengths = []
    while times > 0:
        times = times - 1
        done = False
        ep_reward = 0
        ep_length = 0
        obs = env.reset()
        while not done:
            action, _states = model.predict(observation=obs, deterministic=True)
            #print(action)
            obs, reward, done, truncated, info = env.envs[0].step(action)
            cv2.imshow("img",obs["image"])
            cv2.waitKey(1)
            ep_reward += reward
            ep_length += 1
            if done:
                print("Episode:{}, Reward:{}".format(20-times,ep_reward))
            # print(reward
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
    env.close()
    avg_reward = sum(episode_rewards) / 20
    avg_length = sum(episode_lengths) / 20

    print("var reward is" + str(np.var(episode_rewards)))
    print("var length is" + str(np.var(episode_lengths)))
    print("avg reward is " + str(avg_reward))
    print("avg length is " + str(avg_length))