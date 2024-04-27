import time
import UnityEnvClass.Unity_Multi_input_CarRace_LidarEnv
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.tqc.tqc import TQC
from utils.custom_extractors import *

from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *
from utils.siamese.BarlowTwins import BarlowTwinsFeatureExtractor
from utils.siamese.VICREGModel import VICRegFeatureExtractor
from utils.siamese.VICREG_AP_Model import VICRegAPFeatureExtractor

if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env_name = "CarRaceLidarRandom"
    Algorithm = "PPO"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm,env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarRace_LidarEnv.UnityBaseEnv, n_envs=5, env_kwargs=dict(file_name="../../UnityEnv/CarRaceLidarRandom/RLEnvironments.exe", time_scale=10,max_steps=2500))
    # policy_kwargs = dict(features_extractor_class=VICRegFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01, weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None)
    #                      )
    policy_kwargs = dict(features_extractor_class=VICRegAPFeatureExtractor,
                         features_extractor_kwargs=dict(base_lr=0.01, weight_decay=1e-6,
                                                        sim_coeff=25.0,
                                                        std_coeff=25.0,
                                                        cov_coeff=1.0,
                                                        ckpt=None, action_space=env.action_space)
                         )
    model = PPO("MultiInputPolicy",env,verbose=1,device="cuda",learning_rate=3e-4,seed=4542,gamma=0.99,policy_kwargs=policy_kwargs,use_siamese=True,adjust_learning_rate=True)


    model.set_logger(new_logger)
    model.learn(total_timesteps=10000000, log_interval=4,callback=TensorboardCallback())