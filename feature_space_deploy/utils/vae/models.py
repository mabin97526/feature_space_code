import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.custom_networks import *
from torch.nn.modules.activation import MultiheadAttention
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


class RayEncoder(nn.Module):

    def __init__(self,inputchannel,latent_dim,observation_space,recurrent_size=256):
        super(RayEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(inputchannel, 16, 1, 4, padding=0),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1, 2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.recurrent_size = recurrent_size
        self.n_flatten = self.get_conv_output(observation_space)
        self.gru = nn.GRU(self.n_flatten,recurrent_size,batch_first=True)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.fc_mu = nn.Linear(recurrent_size,latent_dim)
        self.fc_logvar = nn.Linear(recurrent_size,latent_dim)
    def forward(self,x,time_sequence,sequence_length=1):
        x = self.cnn(x)
        if sequence_length == 1:
            x, recurrent_hidden = self.gru(x.unsqueeze(1), time_sequence)
            x = x.squeeze(1)
        else:
            x_shape = tuple(x.size())
            x = x.reshape((x_shape[0] // sequence_length), sequence_length, x_shape[1])
            output, recurrent_hidden = self.gru(x, time_sequence)
            x_shape = tuple(output.size())
            x = x.reshape(x_shape[0] * x_shape[1], x_shape[2])
        return self.fc_mu(x), self.fc_logvar(x), recurrent_hidden
    def init_recurrent_cell_states(self, num_sequences:int, device:torch.device) -> tuple:

        hxs = torch.zeros((num_sequences), self.recurrent_size, dtype=torch.float32, device=device).unsqueeze(0)
        return hxs
    def get_conv_output(self,shape):
        s = torch.zeros(1, *shape)
        o = self.cnn(s)
        return int(np.prod(o.size()))

class RayDecoder(nn.Module):

    def __init__(self, outputchannel, latent_dim, n_flatten,recurrent_size=256):
        super(RayDecoder,self).__init__()
        self.n_lattents = latent_dim
        self.recurrent_size = recurrent_size
        self.fc = nn.Linear(latent_dim, recurrent_size)
        self.gru = nn.GRU(recurrent_size,hidden_size=n_flatten,batch_first=True)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32,16,kernel_size=1,stride=2,padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(16,outputchannel,kernel_size=1,stride=4,padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, time_sequence, sequence_length=1):
        x = self.fc(x)
        x = x.unsqueeze(1)
        x, _ = self.gru(x,time_sequence)
        x = x.squeeze(1)
        x = self.deconv(x)
        return x

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:

        hxs = torch.zeros((num_sequences), self.recurrent_size, dtype=torch.float32, device=device).unsqueeze(0)
        return hxs



class VisualEncoder(nn.Module):

    def __init__(self,inputchannel,latent_dim ,observation_space,recurrent_size=256):
        super(VisualEncoder,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(inputchannel, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.recurrent_size = recurrent_size

        self.n_flatten = self.get_conv_output(observation_space)
        self.gru = nn.GRU(self.n_flatten,recurrent_size,batch_first=True)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.fc_mu = nn.Linear(recurrent_size,latent_dim)
        self.fc_logvar = nn.Linear(recurrent_size,latent_dim)

    def forward(self,x,time_sequence,sequence_length=1):
        x = self.cnn(x)
        if sequence_length == 1 :
            x, recurrent_hidden = self.gru(x.unsqueeze(1),time_sequence)
            x = x.squeeze(1)
        else:
            x_shape = tuple(x.size())
            x = x.reshape((x_shape[0]//sequence_length ),sequence_length,x_shape[1])
            output, recurrent_hidden = self.gru(x,time_sequence)
            x_shape = tuple(output.size())
            x = x.reshape(x_shape[0]*x_shape[1],x_shape[2])
        return self.fc_mu(x),self.fc_logvar(x),recurrent_hidden

    def init_recurrent_cell_states(self, num_sequences:int, device:torch.device) -> tuple:

        hxs = torch.zeros((num_sequences), self.recurrent_size, dtype=torch.float32, device=device).unsqueeze(0)
        return hxs

    def get_conv_output(self,shape):
        s = torch.zeros(1, *shape)
        s = s.permute(0,3,1,2)
        o = self.cnn(s)
        return int(np.prod(o.size()))


class VisualDecoder(nn.Module):

    def __init__(self, outputchannel, latent_dim, n_flatten,recurrent_size=256):
        super(VisualDecoder,self).__init__()
        self.n_lattents = latent_dim
        self.recurrent_size = recurrent_size
        self.fc = nn.Linear(latent_dim, recurrent_size)
        self.gru = nn.GRU(recurrent_size,hidden_size=n_flatten,batch_first=True)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32,outputchannel,kernel_size=8,stride=4,padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, time_sequence, sequence_length=1):
        x = self.fc(x)
        x = x.unsqueeze(1)
        x, _ = self.gru(x,time_sequence)
        x = x.squeeze(1)
        x = self.deconv(x)
        return x

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:

        hxs = torch.zeros((num_sequences), self.recurrent_size, dtype=torch.float32, device=device).unsqueeze(0)
        return hxs


class GlobalEncoder(nn.Module):

    def __init__(self,input_dim, latent_dim):
        super(GlobalEncoder,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU()
        )
        self.out1 = nn.Linear(512,latent_dim)
        self.out2 = nn.Linear(512,latent_dim)

    def forward(self,x):
        x = self.layers(x)
        return self.out1(x),self.out2(x)


class GlobalDecoder(nn.Module):
 

    def __init__(self, latent_dim,out_dim):
        super(GlobalDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.out = nn.Linear(512,out_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.out(x)

class ProductOfExperts(nn.Module):

    def forward(self, mu, logvar,attention_weights, eps=1e-8):
        #var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        #T         = 1. / (var + eps)
        #pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_mu     = torch.sum(mu * attention_weights, dim=0)/torch.sum(attention_weights,dim=0).clamp(min=eps)
        var       = torch.exp(logvar) + eps

        pd_var    = 1. / torch.sum(attention_weights / var, dim=0).clamp(min=eps)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

# Extra Components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def prior_expert(size, use_cuda=False):

    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

class VAE(nn.Module):

    def __init__(self, latent_dim, modal, observation_space,use_cuda=False):
        super(VAE,self).__init__()

        self.latent_dim = latent_dim
        self.modal = modal
        self.use_cuda = use_cuda
        self.observation_space = observation_space

        if self.modal == 'semantic' or self.modal == 'depth':
            self.encoder = VisualEncoder(observation_space[0],latent_dim,observation_space,recurrent_size=256)
            n_flatten = self.encoder.get_conv_output(observation_space)
            self.decoder = VisualDecoder(observation_space[0],latent_dim,n_flatten=n_flatten,recurrent_size=256)
        elif self.modal == 'ray':
            self.encoder = RayEncoder(observation_space[0],latent_dim,observation_space)
            n_flatten = self.encoder.get_conv_output(observation_space)
            self.decoder = RayDecoder(observation_space[0],latent_dim,n_flatten,recurrent_size=256)
        else:
            raise ValueError("Not implemented!")


    def reparametrize(self,mu , logvar):
 
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        if mu.is_cuda:
            epsilon.cuda()
        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()
        # return z = std * epsilon + mu
        return mu.addcmul(std,epsilon)

    def forward(self,x,time_sequence=None,sample=True):
        mu, logvar,recurrent_sequence = self.encoder(x,time_sequence)
        z = self.reparametrize(mu,logvar)
        out = self.decoder(z,recurrent_sequence)
        return out, z,recurrent_sequence,mu,logvar

class ECA_Block(nn.Module):
    def __init__(self,channel, b=1 , gamma=2):
        super(ECA_Block, self).__init__()
        kernel_size = int(abs((math.log(channel,2)+b)/gamma))
        kernel_size = kernel_size if kernel_size%2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size - 1) // 2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)


class MMModel(nn.Module):

    def __init__(self,top_latents,semantic_lantents,depth_latents,ray_latents,attention="eca",use_cuda=False):
        super(MMModel, self).__init__()

        self.use_cuda = use_cuda
        self.n_latents = top_latents
        self.semantic_vae = VAE(latent_dim=semantic_lantents,modal='semantic',observation_space=[84,84,3],use_cuda=use_cuda)
        self.depth_vae    = VAE(latent_dim=depth_latents,modal='depth',observation_space=[84,84,1],use_cuda=use_cuda)
        self.ray_vae      = VAE(latent_dim=ray_latents,modal='ray',observation_space=[400,1],use_cuda=use_cuda)

        self.semantic_top_encoder = GlobalEncoder(input_dim=semantic_lantents,latent_dim=top_latents)
        self.semantic_top_decoder = GlobalDecoder(latent_dim=top_latents,out_dim=semantic_lantents)

        self.depth_top_encoder    = GlobalEncoder(input_dim=depth_latents,latent_dim=top_latents)
        self.depth_top_decoder    = GlobalDecoder(latent_dim=top_latents,out_dim=depth_latents)

        self.ray_top_encoder      = GlobalEncoder(input_dim=ray_latents,latent_dim=top_latents)
        self.ray_top_decoder      = GlobalDecoder(latent_dim=top_latents,out_dim=ray_latents)
        if attention=="eca":
            self.attention = ECA_Block(channel=top_latents*4)
        else:
            self.attention = nn.MultiheadAttention(top_latents*4, 4)
        self.attention_based_poe = ProductOfExperts()

    def reparametrize(self,mu,logvar):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        if mu.is_cuda:
            epsilon = epsilon.cuda()
        std = logvar.mul(0.5).exp_()
        return mu.addcmul(std, epsilon)

    def infer(self, z_semantic,z_depth,z_ray):

        batch_size = z_semantic.size(0)
        use_cuda = next(self.parameters()).is_cuda
        # 1 , batch_size, n_latents
        mu, logvar = prior_expert((1,batch_size,self.n_latents),use_cuda=use_cuda)
        # (1,batch_size,n_latents) == > (2,batch_size,nlatents)
        if z_semantic is not None:
            semantic_mu, semantic_logvar = self.semantic_top_encoder(z_semantic)
            mu                           = torch.cat((mu,semantic_mu.unsqueeze(0)),dim=0)
            logvar                       = torch.cat((logvar,semantic_logvar.unsqueeze(0)),dim=0)
        # (2,batch_size,n_latents) ==> (3,batch_size,n_latents)
        if z_depth is not None:
            depth_mu, depth_logvar       = self.depth_top_encoder(z_depth)
            mu                           = torch.cat((mu,depth_mu.unsqueeze(0)),dim=0)
            logvar                       = torch.cat((logvar,depth_logvar.unsqueeze(0)),dim=0)
        if z_ray is not None:
            ray_mu, ray_logvar           = self.depth_top_encoder(z_ray)
            mu                           = torch.cat((mu,ray_mu.unsqueeze(0)),dim=0)
            logvar                       = torch.cat((logvar,ray_logvar.unsqueeze(0)),dim=0)
        if mu.shape(0) != 4:
            mu_r,logvar_r = prior_expert((4-mu.shape(0),batch_size,self.n_latents),use_cuda=use_cuda)
            mu = torch.cat((mu,mu_r),dim=0)
            logvar = torch.cat((logvar,logvar_r),dim=0)
        attention_weights = self.attention(mu)
        mu, logvar = self.attention_based_poe(mu,logvar,attention_weights)
        return mu, logvar

    def generate(self,semantic=None, depth=None, ray=None):
        with torch.no_grad():
            if semantic is not None:
                semantic_out,semantic_z,semantic_recurrent_sequence,semantic_mu,semantic_logvar = self.semantic_vae(semantic)
            else:
                semantic_out, semantic_mu = None,None
            if depth is not None:
                depth_out,depth_z,depth_recurrent_sequence,depth_mu,depth_logvar = self.depth_vae(depth)
            else:
                depth_out, depth_mu = None,None
            if ray is not None:
                ray_out,ray_z,ray_recurrent_sequence,ray_mu,ray_logvar = self.ray_vae(depth)
            else:
                ray_out, ray_mu = None,None

            top_mu, top_logvar = self.infer(z_semantic=semantic_mu,z_depth=depth_mu,z_ray=ray_mu)
            top_z              = self.reparametrize(top_mu,top_logvar)

            semantic_z = self.semantic_top_decoder(top_z)
            semantic_top_out = self.semantic_vae.decoder(semantic_z)

            depth_z = self.depth_top_decoder(top_z)
            depth_top_out = self.depth_vae.decoder(depth_z)

            ray_z = self.ray_top_decoder(top_z)
            ray_top_out = self.ray_vae.decoder(ray_z)

        return [semantic_out, semantic_top_out],[depth_out,depth_top_out],[ray_out,ray_top_out]
    def gen_latent(self, semantic=None, depth=None,ray=None):
        with torch.no_grad():
            # Encode MNIST Modality Data
            if semantic is not None:
                _,_, _, semantic_mu, _ = self.semantic_vae(semantic)
            else:
                semantic_mu = None

            # Encode Label Modality Data
            if depth is not None:
                _,_,_,depth_mu,_ = self.depth_vae(depth)
            else:
                depth_mu = None
            if ray is not None:
                _,_,_,ray_mu,_ = self.ray_vae(ray)
            else:
                ray_mu = None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_semantic=semantic_mu,z_depth=depth_mu,z_ray=ray_mu)

        return top_mu

    def encode_latent(self, semantic=None, depth=None, ray=None):
        with torch.no_grad():

            # Encode MNIST Modality Data
            if semantic is not None:
                _, _,_ ,semantic_mu, _ = self.semantic_vae(semantic)
            else:
                semantic_mu = None

            # Encode Label Modality Data
            if depth is not None:
                _, _, _,depth_mu, _ = self.depth_vae(depth)
            else:
                depth_mu = None
            if ray is not None:
                _, _, _,ray_mu, _ = self.ray_vae(ray)
            else:
                ray_mu = None

            # Encode top representation
            top_mu, top_logvar = self.infer(z_semantic=semantic_mu, z_depth=depth_mu, z_ray = ray_mu)
            top_z = self.reparametrize(top_mu, top_logvar)

        return top_z, top_mu, top_logvar

    def forward(self,x_semantic, x_depth, x_ray):

        semantic_out, semantic_z, semantic_recurrent, semantic_mu, semantic_logvar = self.semantic_vae(x_semantic)
        depth_out, depth_z, depth_recurrent, depth_mu, depth_logvar                = self.depth_vae(x_depth)
        ray_out, ray_z, ray_recurrent, ray_mu, ray_logvar                          = self.ray_vae(x_ray)

        top_mu, top_logvar = self.infer(z_semantic=semantic_z.clone().detach(), z_depth=depth_z.clone().detach(),z_ray=ray_z.clone().detach())
        top_z = self.reparametrize(mu=top_mu,logvar=top_logvar)

        semantic_top_z = self.semantic_top_decoder(top_z)
        depth_top_z    = self.depth_top_decoder(top_z)
        ray_top_z      = self.ray_top_decoder(top_z)

        return semantic_out,depth_out,ray_out,[semantic_mu,semantic_logvar,semantic_z,semantic_top_z,semantic_recurrent], \
               [depth_mu,depth_logvar,depth_z,depth_top_z,depth_recurrent],[ray_mu,ray_logvar,ray_z,ray_top_z,ray_recurrent],\
               top_mu, top_logvar



def prior_expert(size, use_cuda=False):
    mu      = Variable(torch.zeros(size))
    logvar  = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


if __name__ == '__main__':
    dict = {}
    str1 = "key"
    str2 = "value"
    dict.update({str1:str2})
    print(dict[str1])