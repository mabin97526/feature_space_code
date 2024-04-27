import datetime
import json
import os
import time
from random import random

import cv2
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from PPO_Algo.replayerbuffer import ReplayBuffer
from PPO_Algo.UnityEnv import  *
from PPO_Algo.normalization import Normalization,RewardScaling
from PPO_Algo.ppo_continuous import PPO_continuous
import numpy as np


def evaluate_policy(args,env,agent,state_norm):
    times=3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        #s = s[0]
        s = agent.check_state(s[0])
        if args.use_state_norm:
            s = state_norm(s,update=False)
        done = False
        episode_reward = 0
        while not done:

            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2*(a-0.5)*args.max_action
            else:
                action = a
            s_,r,done,_ = env.step(action)
            #s_ = agent.check_state(s_[0])
            s_ = agent.check_state(s_[0])
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    return evaluate_reward/times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--env",type=str,help="unityEnv",default="./UnityEnv/4-9")
    parser.add_argument("--max_train_steps",type=int,default=int(9e9),help="Maximum number of steps")
    parser.add_argument("--evaluate_freq",type=float,default=5e3)
    parser.add_argument("--policy_dist",type=str,default="Gaussian")
    parser.add_argument("--batch_size",type=int,default=40000)
    parser.add_argument("--mini_batch_size",type=int,default=256)
    parser.add_argument("--hidden_width",type=int,default=512)
    parser.add_argument("--lr_a",type=float,default=3e-4)
    parser.add_argument("--lr_c",type=float,default=3e-4)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=1e-5, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--env_name", type=str, default="605")
    parser.add_argument("--use_sensorfusion", type=bool, default=False, help="use sensorfusion")
    parser.add_argument("--checkpoint_path", type=str, default=None
                        )
    parser.add_argument("--state_dim", type=int, default=512)
    parser.add_argument("--use_curiosity", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--use_vae", type=bool, default=False)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--algorithm",type=str,default="vanilla ppo")

    opt = parser.parse_args()
    writer = SummaryWriter(
        log_dir='runs/PPO_continuous/{}env_{}_{}_seed_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                 opt.env_name, opt.policy_dist, opt.seed))
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #初始化Unity环境
    env = UnityWrapper(opt.env,worker_id=0,seed=opt.seed,no_graphics=False)
    #Unity环境参数
    opt.action_dim,opt.max_action,opt.max_episode_steps = env.action_space.shape[0],float(env.action_space.high[0]),env._max_episode_steps
    print("state_dim={}".format(opt.state_dim))
    print("action_dim={}".format(opt.action_dim))
    print("max_action={}".format(opt.max_action))
    print("max_episode_steps={}".format(opt.max_episode_steps))
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    total_steps = 0
    agent = PPO_continuous(opt)
    replay_buffer = ReplayBuffer(opt)
    state_norm = Normalization(shape=opt.state_dim)  # Trick 2:state normalization
    if opt.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif opt.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=opt.gamma)
    update_num = 0
    num_episode = 1
    evaluate_rewards = []
    while total_steps < opt.max_train_steps:
        s = env.reset()
        s = agent.check_state(s[0])
        #s = agent.check_state(s[0])
        if opt.use_state_norm:
            # print("shape:",s.shape)
            s = state_norm(s)
        if opt.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        episode_reward = 0
        num_episode += 1
        while not done:
            episode_steps += 1
            #print(s.shape)
            a, a_logprob = agent.choose_action(s)
            if opt.policy_dist == "Beta":
                action = 2 * (a-0.5) * opt.max_action
            else:
                action = a
            s_, r, done, _ = env.step(action)
            #cv2.imshow('img',cv2.cvtColor(s_[0],cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
            #s_ = agent.check_state(s_[0])
            s_ = s_[0]
            s_ = agent.check_state(s_)
            if opt.use_state_norm:
                # print("next_sshape:",s_.shape)
                s_ = state_norm(s_)
            if opt.use_reward_norm:
                r = reward_norm(r)
            elif opt.use_reward_scaling:
                r = reward_scaling(r)
            episode_reward += r
            #s_ = s_.reshape([21168])
            if done :
                print(episode_steps)
                writer.add_scalar('episode_rewards_{}'.format(opt.env_name), episode_reward, num_episode)

                print("The {}th episode reward is :{}".format(num_episode, episode_reward))
                if episode_steps != opt.max_episode_steps:
                    dw = True
            else:
                dw = False
            #s_s = s.reshape([21168])
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            #s = s_.reshape(84,84,3)
            total_steps += 1
        if replay_buffer.count == opt.batch_size:
            print("total_steps=", total_steps)
            agent.update(replay_buffer, total_steps, writer, update_num)
            update_num += 1
            replay_buffer.count = 0
            print("Update Model !")
        if num_episode %100 == 0:
            print("evaluate!")
            eva_r = evaluate_policy(opt,env,agent,state_norm)
            evaluate_rewards.append(eva_r)
            print("evaluate reward is ",eva_r)
            writer.add_scalar("evaluate_rewards",evaluate_rewards[-1],global_step=total_steps)
        if update_num % 10  == 0 and update_num != 0:
            agent.save_checkpoint(opt.algorithm,update_num)

