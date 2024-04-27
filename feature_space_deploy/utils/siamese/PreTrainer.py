import argparse
import datetime
import math
import os

import numpy as np
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from utils.siamese.BarlowModel import BarlowTwinsFeatureExtractor
from utils.vae.data_generator import MultiModalDataSet
from gymnasium import spaces

import UnityEnvClass.Unity_Multi_input_CarRace_Env
from dataset import data_utils
import torch
from PIL import Image
def make_dataloaders(
        train_dataset_samples,test_dataset_samples,seed,batch_size,path):
    #total_samples = train_dataset_samples+test_dataset_samples
    dataset = MultiModalDataSet(path)
    ds_train,ds_test = torch.utils.data.random_split(dataset,
                                                     [train_dataset_samples,test_dataset_samples])
    #print(list(ds_test))
    train_loader = torch.utils.data.DataLoader(ds_train,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test,batch_size=batch_size,shuffle=True)

    return train_loader,test_loader

class Trainer:
    def __init__(self,args,model):
        self.model = model
        self.use_cuda = args.usecuda
        if self.use_cuda:
            self.model.cuda()
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        #self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optim = self.model.optimizer
        self.steps = 0
        self.transform = T.Compose([
            T.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),

            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = T.Compose([
            T.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),

            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    def loss_function(self,y,y_):
        return self.model.compute_loss(y,y_)

    def train(self,
              epochs,
              train_dataloader,
              test_dataloader,
              cuda,
              env,writer):
        scaler = torch.cuda.amp.GradScaler()
        for i in range(epochs):
            is_training = True
            self._run(is_training, i, train_dataloader,
                                   cuda,writer,scaler)
            if(i%100==0 and i!=0):
                is_training = False
                self._run(is_training,i,test_dataloader,cuda,writer,scaler)

            if(i%100==0 and i!=0):
                self.save_checkpoint({
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                }, env, i)


    def _run(self, train, epoch, dataloader, cuda,writer,scaler):
        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Test'
            self.model.eval()
        losses = []
        #batch_idx = 0

        for batch_idx,(rgbs,semantics, rays,dones) in enumerate(dataloader):
            #print(batch_idx)
            if cuda:
                dones,rays,rgbs,semantics = dones.cuda(),rays.cuda(),rgbs.cuda(),semantics.cuda()

            if train:
                self.optim.zero_grad()
            img_1 = self.transform(rgbs)
            img_1_aug = self.transform_prime(rgbs)
            semantic_1 = self.transform(semantics)
            semantic_1_aug = self.transform_prime(semantics)
            obs1 = {"image":img_1,"semantic":semantic_1}
            obs1_ = {"image":img_1_aug,"semantic":semantic_1_aug}
            adjust_learning_rate(3000,batch_size=self.batch_size,learning_rate_weights=0.2,learning_rate_biases=0.0048,optimizer=self.optim,loader=dataloader,step=self.steps)
            loss1 = self.loss_function(obs1,obs1_)
            loss = loss1
            self.steps+=1
            losses.append(loss.item())
            if train:
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
            if batch_idx % 10 == 0:
                print(f'{str_name} Epoch: {epoch}'
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss:{np.mean(losses):.6f}')
            batch_idx+=1
        writer.add_scalar("loss",np.mean(losses),epoch)
    def save_checkpoint(self,state, env_name, suffix="", ckpt_path=None):

        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/extractor_checkpoint_{}_{}.pth.tar".format(env_name, suffix)
            #print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save(state, ckpt_path)

def adjust_learning_rate(epochs,batch_size, learning_rate_weights,learning_rate_biases,optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * learning_rate_biases

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for extractor networks")
    parser.add_argument("--epochs", type=int, default=3000, help=" Maximum number of training epochs")
    parser.add_argument("--usecuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--train_dataset_samples", type=int, default=1200)
    parser.add_argument("--test_dataset_samples", type=int, default=390)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--path",type=str,default="./mult-modal_325.pt")
    args = parser.parse_args()
    train_loader, test_loader = make_dataloaders(
        args.train_dataset_samples, args.test_dataset_samples, seed=42, batch_size=args.batch_size,path=args.path
    )

    model = None
    env = UnityEnvClass.Unity_Multi_input_CarRace_Env.UnityBaseEnv(file_name="../../UnityEnv/CarRace/",worker_id=1)
    observation_space = env.observation_space
    env.close()
    for key,subspace in observation_space.spaces.items():
        if len(subspace.shape) > 1 :
            height, width, channels = subspace.shape
            new_shape = (channels, height, width)
            observation_space[key] = spaces.Box(low=0, high=255, shape=new_shape, dtype=subspace.dtype)
    writer = SummaryWriter(log_dir='runs/{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), model))
    model = BarlowTwinsFeatureExtractor(observation_space,lambd=0.0051)
    Trainer = Trainer(args,model)
    Trainer.train(args.epochs,train_loader,test_loader,args.usecuda,"barlowTw",writer)
