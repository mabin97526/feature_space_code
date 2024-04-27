import time
import UnityEnvClass.Unity_Multi_input_CarSearch_InfraredEnv
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
    env_name = "CarRaceNoInfraredSemantic"
    Algorithm = "REDQ"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm,env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarSearch_InfraredEnv.UnityBaseEnv, n_envs=5, env_kwargs=dict(file_name="../../UnityEnv/CarRaceInfrared/RLEnvironments.exe", time_scale=10,max_steps=2000))
    
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
                         features_extractor_kwargs=dict(use_resnet=False),
                         n_critics=4
                         )
    #policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(use_resnet=False))
    
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False),
    #                      n_critics=4, n_quantiles=25
    #                      )
    #model = PPO("MultiInputPolicy",env,verbose=1,device="cuda",learning_rate=3e-4,seed=42,gamma=0.99,policy_kwargs=policy_kwargs)
    #model = SAC("MultiInputPolicy",env,learning_rate=3e-4,buffer_size=20000,batch_size=3000,learning_starts=3000,verbose=1,seed=3998,policy_kwargs=policy_kwargs)
    # td3action_noise
    #action_noise = NormalActionNoise(mean=np.zeros(2),sigma=0.1*np.ones(2))
    #model = TD3("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=4000, batch_size=3000,policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(5, "step"),action_noise=action_noise,verbose=1, seed=7728,policy_delay=2,target_policy_noise=0.3)
    #model = TQC("MultiInputPolicy",env,learning_rate=3e-4,buffer_size=20000,learning_starts=5000,batch_size=3000,train_freq=(1,"step"),top_quantiles_to_drop_per_net=2,policy_kwargs=policy_kwargs,seed=2356)
    model = REDQ("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=5000, batch_size=3000,
                 policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "step"),
                 verbose=1, seed=3322
                 )
    model.set_logger(new_logger)
    model.learn(total_timesteps=300000, log_interval=4,callback=TensorboardCallback())
    env.close()