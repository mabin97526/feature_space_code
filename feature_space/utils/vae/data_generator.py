import os

from torch.utils.data.dataset import Dataset
import torch
import keyboard
from UnityEnvClass.UnityEnvWrapper import UnityWrapper
import numpy as np
import cv2

class MultiModalDataSet(Dataset):
    def __init__(self,data_file):
        self.rgbs,self.semantic,self.rays,self.dones = torch.load(data_file)

        self.semantic = self.semantic.permute(0,3,1,2)
        self.rgbs = self.rgbs.permute(0,3,1,2)
        #self.depth = self.depth.permute(0,3,1,2)
    def __getitem__(self, item):
        rgb,semantic, ray,dones = self.rgbs[item],self.semantic[item],self.rays[item],self.dones[item]
        return rgb,semantic, ray,dones
    def __len__(self):
        return len(self.semantic)

if __name__ == '__main__':
    x_value = 0
    y_value = 0
    #collect data
    env = UnityWrapper(None,seed=44)
    episode = 30
    rgb_data = []
    semantic_data = []
    ray_data = []
    done_data = []
    env.reset()
    while episode > 0:
        if keyboard.is_pressed('w'):
            y_value = min(1, y_value + 0.1)
        elif keyboard.is_pressed('s'):
            y_value = max(-1, y_value - 0.1)
        else:
            y_value = 0

        if keyboard.is_pressed('a'):
            x_value = max(-1, x_value - 0.1)
        elif keyboard.is_pressed('d'):
            x_value = min(1, x_value + 0.1)
        else:
            x_value = 0
        action = (x_value,y_value)
        next_state,reward,done,_ = env.step(action)
        rgb_data.append(next_state[0])
        semantic_data.append(next_state[1])
        ray_data.append(next_state[2])
        cv2.imshow("sem",next_state[1])
        cv2.waitKey(1)
        if done:
            episode -= 1
            x_value = 0
            y_value = 0
            done_data.append(1)
            env.reset()
        else:
            done_data.append(0)
    rgb_data = torch.from_numpy(np.stack(rgb_data)).float()
    #depth_data = torch.from_numpy(np.stack(depth_data)).float()
    semantic_data = torch.from_numpy(np.stack(semantic_data)).float()
    ray_data = torch.from_numpy(np.stack(ray_data)).float()
    done_data = torch.from_numpy(np.stack(done_data))
    if not os.path.exists('dataset/'+'env'):
        os.makedirs('dataset/'+'env')
    data_set = 'dataset/env/mult-modal_325.pt'
    with open(data_set,'wb') as f:
        torch.save((rgb_data,semantic_data,ray_data,done_data),f)
    env.close()
