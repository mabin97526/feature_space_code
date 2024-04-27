
import datetime
import glob
import os
import time
import cv2
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from PPO_Algo.replayerbuffer import ReplayBuffer
from PPO_Algo.UnityEnv import *
from PPO_Algo.normalization import Normalization, RewardScaling
from PPO_Algo.ppo_continuous import PPO_continuous
from nerf.provider import rand_poses, nerf_matrix_to_ngp
from nerf.utils import seed_everything, PSNRMeter, LPIPSMeter, get_rays, linear_to_srgb
import numpy as np

from test_model import test_step


def log(*args, **kwargs):
    print(*args, **kwargs)



def load_checkpoint(checkpoint=None, model_only=False):
    if checkpoint is None:
        checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{opt.name}_ep*.pth'))
        if checkpoint_list:
            checkpoint = checkpoint_list[-1]
            log(f"[INFO] Latest checkpoint is {checkpoint}")
        else:
            log("[WARN] No checkpoint found,model randomly initialized.")
            return
    print(checkpoint)
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    if 'model' not in checkpoint_dict:
        model.load_state_dict(checkpoint_dict)
        log("[INFO] loaded model")
        return
    missing_keys , unexpected_keys = model.load_state_dict(checkpoint_dict['model'],strict=False)
    log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        log(f"[WARN] unexpected keys: {unexpected_keys}")
    if ema is not None and 'ema' in checkpoint_dict:
        ema.load_state_dict(checkpoint_dict['ema'])
    if model.cuda_ray:
        if 'mean_count' in checkpoint_dict:
            model.mean_count = checkpoint_dict['mean_count']
        if 'mean_density' in checkpoint_dict:
            model.mean_density = checkpoint_dict['mean_density']
    if model_only:
        return
    stats = checkpoint_dict['stats']
    epoch = checkpoint_dict['epoch']
    global_step = checkpoint_dict['global_step']
    log(f"[INFO] load at epoch {epoch},global step {global_step}")

    if optimizer and 'optimizer' in checkpoint_dict:
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            log("[INFO] loaded optimizer")
        except:
            log("[WARN] Failed to load optimizer")
    if lr_scheduler and 'lr_scheduler' in checkpoint_dict:
        try:
            lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
            log("[INFO] loaded scheduler")
        except:
            log("[WARN] Failed to load scheduler")
    if scaler and 'scaler' in checkpoint_dict:
        try:
            scaler.load_state_dict(checkpoint_dict['scaler'])
            log("[INFO] loaded scaler")
        except:
            log("[WARN] Failed to load scaler.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    #network backbone options
    parser.add_argument('--fp16', action='store_true',help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument('--write_video', action='store_true',help="whether to store video ")
    parser.add_argument('--down_scale',type=int,default=1,help="downscale of dataset")
    parser.add_argument('--name',type=str,default='ngp',help="experiment name")
    parser.add_argument('--env',type=str ,default = '../UnityEnv/2023_2_28ENV/605.exe',help="UnityENV")



    ###PPO算法参数
    parser.add_argument("--max_train_steps", type=int, default=int(9e9), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=512,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--env_name", type=str, default="605")
    parser.add_argument("--use_sensorfusion", type=bool, default=False, help="use sensorfusion")
    parser.add_argument("--checkpoint_path", type=str, default=None
                        )
    parser.add_argument("--state_dim", type=int, default=21168)
    parser.add_argument("--use_curiosity", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--use_vae", type=bool, default=False)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--use_compose",type =bool ,default=False)
    opt = parser.parse_args()

    writer = SummaryWriter(
        log_dir='runs/PPO_continuous/{}env_{}_{}_seed_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                 opt.env_name, opt.policy_dist, opt.seed))
    #初始化nerf网络
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    seed_everything(opt.seed)


    model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        )
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='none')
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    model.to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    ema = None
    scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)
    #ckpt_path = os.path.join(opt.workspace, 'checkpoints')
    load_checkpoint(opt.ckpt)
    model.eval()
    #相机内参
    '''W = 85
    H = 85
    cx = 128/opt.down_scale
    cy = 128/opt.down_scale
    fl_x = 221.7025033688163/opt.down_scale
    fl_y = 221.7025033688163/opt.down_scale'''
    W = 84
    H = 84
    cx = 42 / opt.down_scale
    cy = 42 / opt.down_scale
    fl_x = 72.74613391789285 / opt.down_scale
    fl_y = 72.74613391789285 / opt.down_scale
    intrinsics = np.array([fl_x, fl_y, cx, cy])

    rand_pose = rand_poses(1, device, radius=1)
    radius = 1
    rays = get_rays(rand_pose, intrinsics, H, W, -1)
    data = {
        'H': H,
        'W': W,
        'rays_o': rays['rays_o'],
        'rays_d': rays['rays_d'],
    }
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            preds = test_step(data, model, opt)
        s = preds[0].detach().cpu().numpy()
        s = (s * 255).astype(np.uint8)


    #初始化Unity环境
    env = UnityWrapper("./UnityEnv/2023_3_6/605.exe", worker_id=0, seed=1, no_graphics=False)
    #Unity 环境参数
    opt.action_dim, opt.max_action, opt.max_episode_steps = env.action_space.shape[0], float(
        env.action_space.high[0]), env._max_episode_steps
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
    code = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 25
    video_out = cv2.VideoWriter('result.avi',code,fps,(84,84),isColor=True)
    while total_steps < opt.max_train_steps:
        s = env.reset()

        s = nerf_matrix_to_ngp(s[0].reshape(4,4), scale=opt.scale, offset=opt.offset)
        s = torch.from_numpy(s).unsqueeze(0)
        print(s.shape)
        radius = s[:, :3, 3].norm(dim=-1).mean(0).item()

        rays = get_rays(s, intrinsics, H, W, -1)
        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'].to(device),
            'rays_d': rays['rays_d'].to(device),
        }

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                preds = test_step(data,model,opt)
            s = preds[0].detach().cpu().numpy()
            s = ( s * 255).astype(np.uint8)

        if opt.use_state_norm:
            # print("shape:",s.shape)
            s = state_norm(s)
        if opt.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        episode_reward = 0
        num_episode += 1
        s = s.flatten()
        while not done:
            episode_steps += 1


            a, a_logprob = agent.choose_action(s)
            if opt.policy_dist == "Beta":
                action = 2 * (a - 0.5) * opt.max_action  # [0,1]->[-max,max]
            else:
                action = a
            a[0]+= 0.9
            s_, r, done, _ = env.step(action)
            s_ = nerf_matrix_to_ngp(s_[0].reshape(4, 4), scale=opt.scale, offset=opt.offset)
            s_ = torch.from_numpy(s_).unsqueeze(0)


            radius = s_[:, :3, 3].norm(dim=-1).mean(0).item()
            rays = get_rays(s_, intrinsics, H, W, -1)

            data = {
                'H': H,
                'W': W,
                'rays_o': rays['rays_o'].to(device),
                'rays_d': rays['rays_d'].to(device),
            }
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=opt.fp16):
                    preds = test_step(data,model,opt)
                s_ = preds[0].detach().cpu().numpy()

                s_ = ( s_ * 255).astype(np.uint8)
            cv2.imwrite('./result_{}.png'.format(episode_steps), cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
            '''
            cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            '''
            '''
            video_out.write(cv2.cvtColor(s_,cv2.COLOR_RGB2BGR))
            cv2.imshow('recon',cv2.cvtColor(s_,cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                print("录制结束")
                video_out.release()'''



            if opt.use_state_norm:
                #print("next_sshape:",s_.shape)
                s_ = state_norm(s_)
            if opt.use_reward_norm:
                r = reward_norm(r)
            elif opt.use_reward_scaling:
                r = reward_scaling(r)
            episode_reward += r
            if done :
                dw = True
                writer.add_scalar('episode_rewards_{}'.format(opt.env_name), episode_reward, num_episode)

                print("The {}th episode reward is :{}".format(num_episode, episode_reward))

            else:
                dw = False
            s_ = s_.flatten()
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1
            if replay_buffer.count == opt.batch_size:
                print("total_steps=", total_steps)
                agent.update(replay_buffer, total_steps, writer, update_num)
                update_num += 1
                replay_buffer.count = 0
                print("Update Model !")

