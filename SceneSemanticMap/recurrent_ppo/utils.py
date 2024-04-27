import numpy as np
import torch

from environments.cartpole_env import CartPole
from environments.minigrid_env import Minigrid
from environments.poc_memory_env import PocMemoryEnv
from environments.memory_gym_env import MemoryGymWrapper
from environments.CarRacing_env import CarRacing
from environments.Unity_Base_env import UnityBaseEnv
from tensoRF.network_cc import NeRFNetwork as CCNeRF
def create_env(config:dict, render:bool=True,worker_id=1):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        config {dict}: The configuration of the environment.

    Returns:
        {env}: Returns the selected environment instance.
    """
    if config["type"] == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True)
    if config["type"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["type"] == "CartPoleMasked":
        return CartPole(mask_velocity=True, realtime_mode = render)
    if config["type"] == "Minigrid":

        return Minigrid(env_name = config["name"], realtime_mode = render)
    if config["type"] == "MemoryGym":
        return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode = render)
    if config["type"] == "CarRacing":
        return CarRacing(env_name=config["name"],render=False)
    if config["type"] == "UnityCarEnv" and not config["use_compose"]:
        return UnityBaseEnv(file_name="../UnityEnv/TrafficCar2/",worker_id=worker_id)
    if config["type"] == "UnityCarEnv" and config["use_compose"] and config["exp_name"]=="CarChaseCompose":
        return UnityBaseEnv(file_name="../UnityEnv/CarChaseCompose/",worker_id=worker_id)
    if config["type"] == "UnityCarEnv" and config["use_compose"] and config["exp_name"]=="CarSearchWithObs":
        return UnityBaseEnv(file_name="../UnityEnv/CarSearch_75/",worker_id=worker_id)
    if config["type"] == "UnityCarRace":
        return UnityBaseEnv(file_name="../UnityEnv/CarRace/",worker_id=worker_id)


def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)


def get_EmptyScene(nerf_config,device):
    model = CCNeRF(
        rank_vec_density=[1],
        rank_mat_density=[1],
        rank_vec=[1],
        rank_mat=[1],
        resolution=[1] * 3,  # fake resolution
        bound=nerf_config["bound"],  # a large bound is needed
        cuda_ray=nerf_config["cuda_ray"],
        density_scale=1,
        min_near=nerf_config["min_near"],
        density_thresh=nerf_config["density_thresh"],
        bg_radius=nerf_config["bg_radius"],
    ).to(device)
    return model

def process_state(state, model, obj1=None, obj2=None):
    #s = np.array_split(state, 7)
    s = np.array_split(state,2)
    for r in s:
        if ((r[0] == 0) and (r[1] == 1)):
            model.compose(obj1, s=r[4], t=np.array([r[3], 0, r[2]]))
        elif r[0] == 1 and r[1] == 0:
            model.compose(obj2, s=r[4] + 0.1, t=np.array([r[3], 0, r[2]]))
        else:
            continue
    '''s = np.array_split(state, 3)
    for r in s:
        print(r)
        if ((r[0] == 0) and (r[1] == 1)):
            # print(r[2])
            model.compose(obj1, s=0.5, t=np.array([r[3] - 1, r[6] / 10, (r[2] - 0.5) * 2.2]))
        elif r[0] == 1 and r[1] == 0:
            # print(r[2])
            model.compose(obj2, s=1, t=np.array([r[3] - 1, r[6] / 10, (r[2] - 0.5) * 2.2]))
        else:
            continue'''

def test_step(data, model, bg_color=None, perturb=False, device='cuda'):
    rays_o = data['rays_o']
    ## current:4096 *3 target:7225 * 3
    rays_d = data['rays_d']

    H, W = data['H'], data['W']

    if bg_color is not None:
        bg_color = bg_color.to(device)

    outputs = model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb)

    pred_rgb = outputs['image'].reshape(-1, H, W, 3)
    return pred_rgb

def load_model(path,nerf_config,device):
    checkpoint_dict = torch.load(path, map_location=device)

    model = CCNeRF(
        rank_vec_density=checkpoint_dict['rank_vec_density'],
        rank_mat_density=checkpoint_dict['rank_mat_density'],
        rank_vec=checkpoint_dict['rank_vec'],
        rank_mat=checkpoint_dict['rank_mat'],
        resolution=checkpoint_dict['resolution'],
        bound=nerf_config["bound"],
        cuda_ray=nerf_config["cuda_ray"],
        density_scale=1,
        min_near=nerf_config["min_near"],
        density_thresh=nerf_config["density_thresh"],
        bg_radius=nerf_config["bg_radius"],
    ).to(device)
    model.load_state_dict(checkpoint_dict['model'], strict=False)
    return model