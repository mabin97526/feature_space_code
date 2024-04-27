import time

import cv2

import UnityEnvClass.Unity_Multi_input_CarRace_Env
from UnityEnvClass import Unity_RollerBall, Unity_3Dball, Unity_3DballVision, Unity_Multi_input_CarRace_NormalEnv,Unity_Multi_input_CarRace_Env,Unity_Multi_input_CarRace_LidarEnv,Unity_Multi_input_CarRace_Env
from UnityEnvClass import Unity_Multi_input_CarRace_BoundingBox_SemanticEnv,Unity_Multi_input_CarSearch_Semantic_Env,Unity_Multi_input_CarRace_InfraredEnv,Unity_Multi_input_CarRace_SemanticEnv
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.tqc.tqc import TQC
from utils.custom_extractors import *
import torch as th
from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *
from utils.siamese.VICREG_AP_Model import VICRegAPFeatureExtractor

if __name__ == '__main__':
    # env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarSearch_Semantic_Env.UnityBaseEnv, n_envs=1,start_index=20,
    #                      env_kwargs=dict(file_name="../../UnityEnv/CarSearchSemanticRandom/RLEnvironments.exe", time_scale=1,
    #                                      max_steps=2500))
    env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarRace_SemanticEnv.UnityBaseEnv,n_envs=1,start_index=33,env_kwargs=dict(file_name="../../UnityEnv/RaceDeploy/RLEnvironments.exe", time_scale=1,max_steps=2000))
    # env = make_unity_env(UnityEnvClass.Unity_Multi_input_CarSearch_Semantic_Env.UnityBaseEnv, n_envs=1, start_index=55,
    #                       env_kwargs=dict(file_name="../../UnityEnv/CarSearchSemanticRandom/RLEnvironments.exe", time_scale=1,
    #                                       max_steps=2500))
    #policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                     features_extractor_kwargs=dict(use_resnet=False),
    #                     n_critics=4, n_quantiles=25)
    #policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(use_resnet=False))
    # REDQ
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False),
    #                      n_critics=4
    #                      )
    # policy_kwargs = dict(features_extractor_class=VICRegAPFeatureExtractor,
    #                      features_extractor_kwargs=dict(base_lr=0.01, weight_decay=1e-6,
    #                                                     sim_coeff=25.0,
    #                                                     std_coeff=25.0,
    #                                                     cov_coeff=1.0,
    #                                                     ckpt=None, action_space=env.action_space)
    #                      )
    #model = PPO("MultiInputPolicy",env,verbose=1,device="cpu",learning_rate=3e-4,seed=42,gamma=0.99,policy_kwargs=policy_kwargs,use_siamese=False,adjust_learning_rate=False)
    #model = SAC("MultiInputPolicy",env,learning_rate=3e-4,buffer_size=10000,batch_size=3000,learning_starts=10000,verbose=1,seed=42,policy_kwargs=policy_kwargs)
    #model = TD3("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=5000, batch_size=3000,
    #            policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "step"),
    #            verbose=1, seed=4562, policy_delay=3, target_policy_noise=0.3)
    #model = TQC("MultiInputPolicy",env,learning_rate=3e-4,
    #            buffer_size=20000,learning_starts=5000,batch_size=3000,
    #            train_freq=(1,"step"),
    #            top_quantiles_to_drop_per_net=2,
    #            policy_kwargs=policy_kwargs)
    # model = REDQ("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=5000, batch_size=3000,
    #              policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "step"),
    #              verbose=1, seed=3322
    #              )

    #model.set_parameters(load_path_or_dict="./20240405-165631ppo550.zip",exact_match=True)
    model = PPO.load("./results/bestppo204.zip",env=env,print_system_info=True)
    print("load success")
    times = 10
    episode_rewards = []
    episode_lengths = []
    while times > 0 :
        times = times - 1
        done = False
        ep_reward = 0
        ep_length = 0
        obs = env.reset()
        step = 0
        while not done:
            #if step % 5 ==0:
            #obs["semantic"] = obs["semantic"]/255.0
            #obs["image"] = obs["image"]/255.0
            #obs_tensor,_ = model.policy.obs_to_tensor(obs)
            # action, values, log_probs = model.policy(obs_tensor)
            # action = action.detach().cpu().numpy()
            # action = np.clip(action, env.action_space.low, env.action_space.high)
            print(obs["ray"])
            action, _states = model.predict(observation=obs, deterministic=True)
            #print(action)
            obs, reward, done, truncated, info = env.envs[0].step(action)
            #arr = np.zeros((84,84,3),dtype=np.int)
            #print(np.array_equal(arr,obs["semantic"]))
            cv2.imshow("image",obs["image"])
            #cv2.imshow("semantic",obs["semantic"])
            cv2.waitKey(1)

            #print(obs["semantic"])
            ep_reward += reward
            ep_length += 1
            step+=1
            # print(reward
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(ep_reward)
    env.close()
    avg_reward = sum(episode_rewards)/10
    avg_length = sum(episode_lengths)/10

    print("var reward is" + str(np.var(episode_rewards)))
    print("var length is" + str(np.var(episode_lengths)))
    print("avg reward is "+ str(avg_reward))
    print("avg length is "+ str(avg_length))
