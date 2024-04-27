import json

import cv2
import numpy as np
import pickle
import torch
from docopt import docopt
import tqdm
from gym.spaces import Box, Tuple as TupleSpace, Tuple
from model import ActorCriticModel
from nerf.provider import nerf_matrix_to_ngp
from nerf.utils import get_rays
from recurrent_ppo.environments.Unity_Base_env import UnityBaseEnv
from utils import create_env,process_state,test_step,get_EmptyScene,load_model
from yaml_parser import YamlParser
from trainer import PPOTrainer
def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/minigrid.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))
    print(config)
    # Instantiate environment
    if config["use_compose"]:
        nerf_config = YamlParser("./configs/CCNeRF.yaml").get_config()
        cube = load_model(path="./composemodel/cube_feature.pth",nerf_config=nerf_config,device=device)
        #ball = load_model(path="../trial_capsule/checkpoints/12_0-12_0.pth", nerf_config=nerf_config, device=device)
        #cube = load_model(path="../trial_stair/checkpoints/12_0-12_0.pth",nerf_config=nerf_config,device=device)

        ball = load_model(path="./composemodel/ball_feature.pth",nerf_config=nerf_config,device=device)

        n_model = get_EmptyScene(nerf_config,device)
        n_model.eval()
        intrinsics = np.array([nerf_config["fl_x"], nerf_config["fl_y"], nerf_config["cx"], nerf_config["cy"]])
        transform_path = nerf_config["path"]
        with open(transform_path, 'r') as f:
            transform = json.load(f)
        frames = transform["frames"]
        poses = []
        for f in tqdm.tqdm(frames,desc=f'Loading test data'):
            pose = np.array(f['transform_matrix'], dtype=np.float32)
            pose = nerf_matrix_to_ngp(pose, scale=nerf_config["scale"], offset=nerf_config["offset"])
            poses.append(pose)
        poses = torch.from_numpy(np.stack(poses,axis=0)).to(device)
        radius = poses[:,:3,3].norm(dim=-1).mean(0).item()
        rays = get_rays(poses,intrinsics,nerf_config["H"],nerf_config["W"],-1)
        data = {
            'H':nerf_config["H"],
            'W':nerf_config["W"],
            'rays_o':rays['rays_o'],
            'rays_d':rays['rays_d'],
        }
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=nerf_config["fp16"]):
                preds = test_step(data, n_model)
            s = preds[0].detach().cpu().numpy()
            s = (s * 255).astype(np.uint8)
        print("load compose model success")
    #env = UnityBaseEnv(file_name='../UnityEnv/CarSearchNoObs_Cube_Compose',worker_id=100)
    env = UnityBaseEnv(file_name='../UnityEnv/CarChaseCompose',worker_id=0)
    action_space_shape = (env.action_space.shape[0],) if config["continuous"] else (env.action_space.n,)
    # Initialize model and load its parameters
    obs_shape = env.observation_space
    model = ActorCriticModel(config, env.observation_space, action_space_shape)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    times = 100
    episode_rewards = []
    episode_lengths = []
    startfarme = 0
    # Run and render episode
    while times>0:
        times = times-1
        done = False


        # Init recurrent cell
        hxs, cxs = model.init_recurrent_cell_states(1, device)
        if config["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif config["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)

        obs = env.reset()
        while not done:
            # Render environment
            #env.render()
            # Forward model
            if config["use_compose"]:
                n_model = get_EmptyScene(nerf_config,device)
                process_state(state=obs,model=n_model,obj1=cube,obj2=ball)
                n_model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=nerf_config["fp16"]):
                        preds = test_step(data=data,model=n_model)
                    obs = preds[0].detach().cpu().numpy()
                    obs = (obs*255).astype(np.uint8)
                cv2.imshow("obs",obs)
                cv2.waitKey(1)


            policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0),dtype=torch.float32).to(device), recurrent_cell, device, 1)
            # Sample action
            action = []

            if config["continuous"]:
                actions = policy.sample().detach()
                #print(action)
                for a in actions.split(1, 0):
                    action.append(a)

            else:
                for action_branch in policy:
                    action.append(action_branch.sample().item())
            # Step environment
            print(action[0])

            #continuous
            #obs, reward, done, info = env.step(action[0].cpu().numpy())
            #discrete
            obs, reward, done, info = env.step(action[0])

    
        # After done calculate res

        episode_rewards.append(info["reward"])
        episode_lengths.append(info["length"])
        print("Episode length: " + str(info["length"]))
        print("Episode reward: " + str(info["reward"]))

    avg_reward = sum(episode_rewards)/100
    avg_length = sum(episode_lengths)/100
    print("var reward is"+str(np.var(episode_rewards)))
    print("var length is"+ str(np.var(episode_lengths)))
    print("Final reward="+str(avg_reward))
    print("Final avg_length="+str(avg_length))

    env.close()

if __name__ == "__main__":
    main()