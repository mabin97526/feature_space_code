import time


import UnityEnvClass.Unity_Multi_input_CarRace_Env
from UnityEnvClass import Unity_RollerBall, Unity_3Dball, Unity_3DballVision, Unity_Base_env
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure

from utils.siamese.BarlowTwins import BarlowTwinsFeatureExtractor
from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *
from utils.siamese.VICREGModel import VICRegFeatureExtractor
from utils.siamese.VICREG_AP_Model import VICRegAPFeatureExtractor

if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env_name = "CarRaceSemanticRand"
    Algorithm = "PPO"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm,env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarRace_Env.UnityBaseEnv, n_envs=5,start_index=15, env_kwargs=dict(file_name="../../UnityEnv/RaceDeploy/RLEnvironments.exe", time_scale=100,max_steps=2000))
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(use_resnet=False))
    #policy_kwargs = dict(features_extractor_class = BarlowTwinsFeatureExtractor,features_extractor_kwargs=dict(lambd=0.0051,ckpt="../../utils/siamese/checkpoints/extractor_checkpoint_barlowTw_100.pth.tar"))

    # policy_kwargs = dict(features_extractor_class=VICRegFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01, weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None,)
    #                      )
    model = PPO("MultiInputPolicy",env,verbose=1,device="cuda",learning_rate=3e-4,seed=42,gamma=0.99,policy_kwargs=policy_kwargs,use_siamese=False,adjust_learning_rate=False,save_reward=9.0)

    model.set_logger(new_logger)
    model.learn(total_timesteps=500000000, log_interval=1,callback=TensorboardCallback())