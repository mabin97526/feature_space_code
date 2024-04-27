import time
import UnityEnvClass.Unity_PushBlock_env
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
    env_name = "PushBlockSiameseBYOL"
    Algorithm = "PPO"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm, env_name) + timestamp,
                           ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_PushBlock_env.UnityBaseEnv, n_envs=5,start_index=0, env_kwargs=dict(file_name="../../UnityEnv/PushBlock/UnityEnvironment.exe", time_scale=10,max_steps=1000))
    # BarlowTwins
    policy_kwargs = dict(features_extractor_class=BarlowTwinsFeatureExtractor,
                         features_extractor_kwargs=dict(lambd=0.0051))
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False))
    # vicReg adjust_learning_rate = True
    # policy_kwargs = dict(features_extractor_class = VICRegFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01,weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None)
    #                      )
    # vicReg + action predictor adjust_learning_rate = True
    # policy_kwargs = dict(features_extractor_class=VICRegAPFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01, weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None,action_space=env.action_space)
    #                      )
    # BYOL
    # policy_kwargs = dict(
    #     features_extractor_class = BYOLFeatureExtractor,
    #     features_extractor_kwargs=dict(moving_average_decay=0.99,learning_rate=3e-4,ckpt=None)
    # )
    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda", learning_rate=3e-4, seed=4331, gamma=0.99,
                policy_kwargs=policy_kwargs, use_siamese=False,adjust_learning_rate=False)
    model.set_logger(new_logger)
    model.learn(total_timesteps=50000000, log_interval=20, callback=TensorboardCallback())