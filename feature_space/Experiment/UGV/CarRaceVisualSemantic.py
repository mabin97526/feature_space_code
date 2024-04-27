import time
import UnityEnvClass.Unity_Multi_input_CarSearch_Visual_Semantic_Env
from stable_baselines3 import SAC, PPO, TD3, REDQ, DDPG
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.tqc.tqc import TQC
from utils.custom_extractors import *

from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *



if __name__ == '__main__':
    timestamp = time.strftime("/%Y%m%d-%H%M%S")
    env_name = "CarRaceVisualSemantic"
    Algorithm = "REDQ"
    new_logger = configure("./loggers/{}_{}_logger".format(Algorithm,env_name) + timestamp, ["stdout", "csv", "tensorboard"])
    env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarSearch_Visual_Semantic_Env.UnityBaseEnv, n_envs=5, env_kwargs=dict(file_name="../../UnityEnv/CarRaceVisualSemantic/RLEnvironments.exe", time_scale=10,max_steps=2000))
    #REDQ
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False),
    #                      n_critics=4
    #                      )
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(use_resnet=False))
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False),
    #                      n_critics=4, n_quantiles=25
    #                      )
    #model = PPO("MultiInputPolicy",env,verbose=1,device="cuda",learning_rate=3e-4,seed=42,gamma=0.99,policy_kwargs=policy_kwargs)
    #model = SAC("MultiInputPolicy",env,learning_rate=3e-4,buffer_size=20000,batch_size=3000,learning_starts=3000,verbose=1,seed=3998,policy_kwargs=policy_kwargs)
    # td3action_noise
    #action_noise = NormalActionNoise(mean=np.zeros(1),sigma=0.1*np.ones(1))
    #model = DDPG("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=4000, batch_size=3000,policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "episode"),action_noise=action_noise,verbose=1, seed=7728)
    #model = TQC("MultiInputPolicy",env,learning_rate=3e-4,buffer_size=20000,learning_starts=5000,batch_size=3000,train_freq=(1,"step"),top_quantiles_to_drop_per_net=2,policy_kwargs=policy_kwargs,seed=2356)
    model = REDQ("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=5000, batch_size=3000,
                 policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "step"),
                 verbose=1, seed=3322
                 )
    model.set_logger(new_logger)
    model.learn(total_timesteps=5000000, log_interval=4,callback=TensorboardCallback())