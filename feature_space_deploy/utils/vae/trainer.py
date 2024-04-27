import torch.optim as optim
import torch.nn.functional as F
import torch
from utils.vae.utils import WarmUp, sym_KLD_gaussian, AverageMeter


class VAETrainer(object):
    def __init__(self,model,args,cuda):
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()
        self.learning_rate = args.learning_rate
        self.lambda_s = args.lambda_semantic
        self.lambda_d = args.lambda_depth
        self.lambda_r = args.lambda_ray

        self.beta_s = args.beta_semantic
        self.beta_d = args.beta_depth
        self.beta_r = args.beta_ray
        self.gamma_s = args.gamma_semantic
        self.gamma_r = args.gamma_ray
        self.gamma_d = args.gamma_det
        self.beta = args.beta_top
        self.alpha_fpa = args.alpha_fpa
        self.batch_size = args.batch_size
        self.wup_mod_epochs = args.wup_mod_epochs
        self.wup_top_epochs = args.wup_top_epochs
        self.beta_s_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_s)
        self.beta_r_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_r)
        self.beta_d_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_d)
        self.beta_t_wup = WarmUp(epochs=self.wup_top_epochs, value=self.beta)

        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_loss = 100000000

    def loss_function(self,semantic_data,depth_data,ray_data,semantic_out,depth_out,ray_out,semantic_dist,depth_dist,ray_dist,mu,logvar,fpa_mu,fpa_logvar):
            semantic_recon = self.lambda_s * torch.sum(F.binary_cross_entropy(
                semantic_out.view(semantic_out.size(0),-1),
                semantic_data.view(semantic_data.size(0),-1),
                reduction='none'),dim=-1)
            semantic_prior = self.beta_s * \
                             (-0.5 * torch.sum(1+semantic_dist[1]-semantic_dist[0].pow(2)-semantic_dist[1].exp(),dim=1))
            depth_recon = self.lambda_d * torch.sum(F.binary_cross_entropy(
                depth_out.view(depth_out.size(0), -1),
                depth_data.view(depth_data.size(0), -1),
                reduction='none'), dim=-1)
            depth_prior = self.beta_s * \
                             (-0.5 * torch.sum(1 + depth_dist[1] - depth_dist[0].pow(2) - depth_dist[1].exp(),
                                               dim=1))
            ray_recon = self.lambda_r * torch.sum(F.mse_loss(ray_out.view(ray_out.size(0),-1),
                                                             ray_data.view(ray_data.size(0),-1),
                                                             reduction='none'
                                                             ),dim=-1)
            ray_prior = self.beta_r * \
                        (-0.5 * torch.sum(1 + ray_dist[1] - ray_dist[0].pow(2)-ray_dist[1].exp(),dim=1))


            semantic_top_recon = self.gamma_s * torch.sum(F.mse_loss(semantic_dist[3],semantic_dist[2].clone().detach(),
                                                                     reduction='none'
                                                                     ),dim=-1)
            depth_top_recon = self.gamma_d * torch.sum(F.mse_loss(depth_dist[3],depth_dist[2].clone().detach(),
                                                                  reduction='none'),dim=-1)
            ray_top_recon = self.gamma_r * torch.sum(F.mse_loss(ray_dist[3],ray_dist[2].clone().detach(),
                                                                reduction='none'),dim=-1)
            top_prior = self.beta_t_wup.get() * \
                        (-0.5 * torch.sum(1+logvar - mu.pow(2)-logvar.exp(),dim=1))
            top_fpa = 0.0
            for i in range(len(fpa_mu)):
                top_fpa += self.alpha_fpa * sym_KLD_gaussian(q_mu=mu, q_logvar=logvar, p_mu=fpa_mu[i],
                                                             p_logvar=fpa_logvar[i])
            top_fpa /= len(fpa_mu)

            loss = torch.mean(
                semantic_recon + depth_recon + ray_recon
                + semantic_prior + depth_prior + ray_prior
                + semantic_top_recon + depth_top_recon + ray_top_recon
                + top_prior + top_fpa
            )
            return loss, semantic_recon, depth_recon, ray_recon, \
                semantic_prior, depth_prior, ray_prior, \
                semantic_top_recon, depth_top_recon, ray_top_recon, \
                top_prior, top_fpa

    def _run(self, train, epoch, dataloader, cuda):
        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()
        loss_meter = AverageMeter()
        semantic_recon_meter = AverageMeter()
        semantic_prior_meter = AverageMeter()
        depth_recon_meter = AverageMeter()
        depth_prior_meter = AverageMeter()
        ray_recon_meter = AverageMeter()
        ray_prior_meter = AverageMeter()
        semantic_top_recon_meter = AverageMeter()
        depth_top_recon_meter = AverageMeter()
        ray_top_recon_meter = AverageMeter()
        prior_meter = AverageMeter()
        fpa_meter = AverageMeter()

        for batch_idx,(semantic, depth, ray) in enumerate(dataloader):
            if cuda:
                semantic, depth, ray = semantic.cuda(), depth.cuda(), ray.cuda()
            if train:
                self.optim.zero_grad()
            semantic_out, depth_out, ray_out, semantic_dist, depth_dist, ray_dist, \
            top_mu, top_logvar
            self.model(semantic,depth,ray)