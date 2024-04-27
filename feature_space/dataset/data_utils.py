import numpy as np

import UnityEnvClass.Unity_Base_env
from UnityEnvClass import *
from torch.utils.data.dataset import Dataset
import os
import torch
import random

class UnityDataSet(Dataset):
    def __init__(self,data_file):
        #self.images ,self.rays,self.dets,self.depths,self.semantics = torch.load(data_file)
        #image_data,ray_data,det_data,depth_data,semantic_data)
        self.prev_images,self.prev_rays,self.actions,self.next_images,self.next_rays = torch.load(data_file)
        self.prev_images = self.prev_images.permute(0,3,1,2)
        self.next_images = self.next_images.permute(0,3,1,2)
        #print(self.depths.shape)
        #self.depths = self.depths.permute(0,3,1,2)
        #self.semantics = self.semantics.permute(0,3,1,2)
        #print(self.depths.shape)
    def __getitem__(self, item):
        #img,ray,det,depth,semantic = self.images[item],self.rays[item],self.dets[item],self.depths[item],self.semantics[item]
        prev_img = self.prev_images[item]
        prev_ray = self.prev_rays[item]
        actions = self.actions[item]
        next_img = self.next_images[item]
        next_ray = self.next_rays[item]
        return prev_img,prev_ray,actions,next_img,next_ray
    def __len__(self):
        return len(self.prev_images)





def generateData(maxSteps,env,dataset):
    print("we need {} steps data".format(maxSteps))
    #image_ray_data = []
    '''semantic_data = []
    ray_data = []
    det_data = []
    depth_data = []'''
    prev_image = []
    prev_ray = []
    action = []
    next_image = []
    next_ray = []
    prev_state_r = env.reset()[0]


    for i in range(maxSteps):
        prev_image.append(prev_state_r["image"]/255.0)
        prev_ray.append(prev_state_r["ray"])
        action_c = (random.uniform(0,1.0), random.uniform(-1.0,1.0))
        next_state_n, reward, done, _,_ = env.step(action_c)
        #print(next_state[2].shape)
        action.append(action_c)
        next_image.append(next_state_n["image"]/255.0)
        next_ray.append(next_state_n["ray"])

        #transitions.put({"pre_state":prev_state,"next_state":next_state,"action":action})

        print("Already generate {} data".format(i))
        #print(next_state[0])
        #print(next_state[1])
        if done:
            prev_state_r = env.reset()[0]

        else:
            prev_state_r = next_state_n



    '''image_data = torch.from_numpy(np.stack(image_data)).float()
    ray_data =torch.from_numpy(np.stack(ray_data)).float().squeeze(1)
    det_data = torch.from_numpy(np.stack(det_data)).float().squeeze(1)
    depth_data = torch.from_numpy(np.stack(depth_data)).float().squeeze(1)
    semantic_data = torch.from_numpy(np.stack(semantic_data)).float().squeeze(1)'''
    prev_image = torch.from_numpy(np.stack(prev_image)).float()
    prev_ray = torch.from_numpy(np.stack(prev_ray)).float()
    action = torch.from_numpy(np.stack(action)).float()
    next_image = torch.from_numpy(np.stack(next_image)).float()
    next_ray = torch.from_numpy(np.stack(next_ray)).float()

    #print(depth_data.shape)
    #print(semantic_data.shape)
    #print(image_data.shape)
    #print(det_data.shape)
    #print(ray_data.shape)
    with open(dataset,'wb') as f:
        #torch.save((image_data,ray_data,det_data,depth_data,semantic_data),f)
        torch.save((prev_image,prev_ray,action,next_image,next_ray),f)




if __name__ == '__main__':
    env = UnityEnvClass.Unity_Base_env.UnityBaseEnv("../UnityEnv/CarRace",worker_id=10)

    if not os.path.exists('results/'+"env"):
        os.makedirs('results/'+"env")
    generateData(5000,env,'results/env/carRace.pt')
    env.close()
    #data  = RollerBallData('data/balldata.pt')
    #img,ray = data.images,data.rays
    #print(img.shape)
    #data.printdata()
    #print(len(data))
    #image,ray = data.__getitem__(0)
    #print(type(image))
    #print(ray)
