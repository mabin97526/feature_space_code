import argparse
import datetime
import os

from torch import optim
from torch.utils.tensorboard import SummaryWriter

import UnityEnvClass.Unity_Multi_input_CarRace_Env
from custom_networks import *
from dataset import data_utils
import torch

def make_dataloaders(
        train_dataset_samples,test_dataset_samples,seed,batch_size,path):
    total_samples = train_dataset_samples+test_dataset_samples
    dataset = data_utils.UnityDataSet(path)
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
    def loss_function(self,y,y_):
        return self.model.compute_loss(y,y_)

    def train(self,
              epochs,
              train_dataloader,
              test_dataloader,
              cuda,
              env,writer):
        info =  {}
        for i in range(epochs):
            is_training = True
            self._run(is_training, i, train_dataloader,
                                   cuda,writer)
            if(i%100==0 and i!=0):
                is_training = False
                self._run(is_training,i,test_dataloader,cuda,writer)

            if(i%100==0 and i!=0):
                self.save_checkpoint({
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                }, env, i)


    def _run(self, train, epoch, dataloader, cuda,writer):
        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Test'
            self.model.eval()
        losses = []
        for batch_idx,(prev_img,prev_semantic,prev_ray,actions,next_img,next_semantic,next_ray) in enumerate(dataloader):
            #print(batch_idx)
            if cuda:
                prev_img,prev_semantic,prev_ray,actions,next_img,next_semantic,next_ray = prev_img.cuda(),prev_semantic.cuda(),prev_ray.cuda(),actions.cuda(),next_img.cuda(),next_semantic.cuda(),next_ray.cuda()

            if train:
                self.optim.zero_grad()
            input1 = dict(image=prev_img,ray=prev_ray)
            input2 = dict(image=next_img,ray = next_ray)
            out = self.model.predict(input1,input2)

            loss = self.loss_function(out,actions)
            losses.append(loss.item())
            if train:

                loss.backward()
                self.optim.step()
            if batch_idx % 10 == 0:
                print(f'{str_name} Epoch: {epoch}'
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss:{np.mean(losses):.6f}')
        writer.add_scalar("loss",np.mean(losses),epoch)
    def save_checkpoint(self,state, env_name, suffix="", ckpt_path=None):

        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/extractor_checkpoint_{}_{}.pth.tar".format(env_name, suffix)
            #print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save(state, ckpt_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for extractor networks")
    parser.add_argument("--epochs", type=int, default=3000, help=" Maximum number of training epochs")
    parser.add_argument("--usecuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--train_dataset_samples", type=int, default=4000)
    parser.add_argument("--test_dataset_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model",type=str,default="transformer")
    parser.add_argument("--path",type=str,default="../dataset/results/env/carRace.pt")
    args = parser.parse_args()
    train_loader, test_loader = make_dataloaders(
        args.train_dataset_samples, args.test_dataset_samples, seed=42, batch_size=args.batch_size,path=args.path
    )

    model = None
    env = UnityEnvClass.Unity_Multi_input_CarRace_Env.UnityBaseEnv(file_name="../UnityEnv/CarRace/",worker_id=1)
    observation_space = env.observation_space
    observation_space["image"] = spaces.Box(low=0,high=255,shape=(12,84,84),dtype=np.uint8)
    env.close()
    writer = SummaryWriter(log_dir='runs/{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), model))
    if(args.model=="tensor_fusion"):
        model = TensorFusion(observation_space=observation_space)
    elif(args.model=="low_rank_tensorfusion"):
        model = LowRankTensorFusion(observation_space=observation_space,output_dim=64,rank=16)
    elif(args.model=="transformer"):
        class HyperParams(MULTModel.DefaultHyperParams):
            num_heads = 2
            embed_dim = 256
            output_dim = 64
            all_steps = True
        model = MULTModel(observation_space=observation_space,hyper_params=HyperParams)
    Trainer = Trainer(args,model)
    Trainer.train(args.epochs,train_loader,test_loader,args.usecuda,args.model,writer)
