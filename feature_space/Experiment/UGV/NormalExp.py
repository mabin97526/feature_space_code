import time


import UnityEnvClass.Unity_Multi_input_CarSearchNormal_Env, UnityEnvClass.Unity_Multi_input_CarRace_NormalEnv
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
    env_name = "CarRaceNormal"
    Algorithm = "REDQ"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm,env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarRace_NormalEnv.UnityBaseEnv, n_envs=5, env_kwargs=dict(file_name="../../UnityEnv/CarRace/RLEnvironments.exe", time_scale=10,max_steps=2500))
    #REDQ
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(use_resnet=False),
                         n_critics=4
                         )
    model = REDQ("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=5000, batch_size=3000,
                 policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "step"),
                 verbose=1, seed=3322
                 )
    model.set_logger(new_logger)
    model.learn(total_timesteps=10000000, log_interval=4,callback=TensorboardCallback())