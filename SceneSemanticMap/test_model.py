import json

import torch
import argparse
from PPO_Algo.UnityEnv import *
from nerf.provider import NeRFDataset, nerf_matrix_to_ngp

from nerf.utils import *


def test_step(data,model,opt,bg_color=None,perturb=False,device='cuda'):
    rays_o = data['rays_o']
    ## current:4096 *3 target:7225 * 3
    rays_d = data['rays_d']

    H , W = data['H'], data['W']

    if bg_color is not None:
        bg_color = bg_color.to(device)

    outputs = model.render(rays_o,rays_d,staged=True,bg_color=bg_color,perturb=perturb,**vars(opt))


    pred_rgb = outputs['image'].reshape(-1, H, W ,3)
    return pred_rgb

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

def log( *args, **kwargs):

        console.print(*args, **kwargs)
        if log_ptr:
            print(*args, file=log_ptr)
            log_ptr.flush()  # write immediately to file





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str,help="data_")
    parser.add_argument('--workspace',type=str,default='workspace')
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
    opt = parser.parse_args()
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    console = Console()
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

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
    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    model.to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:1)
    ema = None
    scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)

    #variable init
    epoch = 0
    global_step = 0
    local_step = 0
    stats = {
        "loss": [],
        "valid_loss": [],
        "results": [],
        "checkpoints":[],
        "best_result":None,
    }
    best_mode = 'min'
    log_ptr = None
    if opt.workspace is not None:
        os.makedirs(opt.workspace,exist_ok=True)
        log_path = os.path.join(opt.workspace,f"log_{opt.name}.txt")
        log_ptr = open(log_path, "a+")
        ckpt_path = os.path.join(opt.workspace ,'checkpoints')
        #print(ckpt_path)
        best_path = f"{ckpt_path}/{opt.name}.pth"
        os.makedirs(ckpt_path,exist_ok=True)
    log(f'[INFO] Trainer: {opt.name} | {time_stamp} | {device} | {"fp16" if opt.fp16 else "fp32"} | {opt.workspace}')
    log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
    if opt.workspace is not None:
        if opt.ckpt == "scratch":
            log("[INFO] Training from scratch ...")
        elif opt.ckpt == "latest":
            log("[INFO] Loading latest checkpoint ...")
            load_checkpoint()
        elif opt.ckpt == "latest_model":
            log("[INFO] Loading latest checkpoint (model only) ...")
            load_checkpoint(model_only=True)
        elif opt.ckpt == "best":
            if os.path.exists(best_path):
                log("[INFO] Loading best checkpoint ...")
                load_checkpoint(best_path)
            else:
                log(f"[INFO] {best_path} not found,loading latest ...")
                load_checkpoint(

                )
        else:
            log(f"[INFO] Loading {opt.ckpt}")
            load_checkpoint(opt.ckpt)
    if opt.rand_pose >= 0:
        from nerf.clip_utils import  CLIPLoss
        clip_loss = CLIPLoss(device)
        clip_loss.prepare_text([opt.clip_text])
    #car_trail_workspace\checkpoints\ngp_ep0066.pth
    #trainer初始化完成
    #读取数据
    transform_path = os.path.join(opt.path,'test.json')
    with open(transform_path,'r') as f:
        transform = json.load(f)
    #读取height 与width并downscale
    if 'h' in transform and 'w' in transform:
        H = int(transform['h']) //opt.down_scale
        W = int(transform['w']) //opt.down_scale

    frames = transform["frames"]

    poses = []
    images = []
    for f in tqdm.tqdm(frames ,desc=f'Loading test data'):

        pose = np.array(f['transform_matrix'],dtype=np.float32)


        pose = nerf_matrix_to_ngp(pose,scale = opt.scale,offset=opt.offset)

        poses.append(pose)

    poses = torch.from_numpy(np.stack(poses,axis=0)).to(device)


    radius = poses[:,:3,3].norm(dim=-1).mean(0).item()

    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / opt.down_scale
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / opt.down_scale
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    cx = (transform['cx']/ opt.down_scale) if 'cx' in transform else (W/2)
    cy = (transform['cy']/ opt.down_scale) if 'cy' in transform else (H/2)
    intrinsics = np.array([fl_x,fl_y,cx,cy])
    print(intrinsics)
    #intrinsics = torch.from_numpy(intrinsics).to(device)
    #获取ray
    rays = get_rays(poses,intrinsics,H,W,-1)

    data = {
        'H': H,
        'W': W,
        'rays_o': rays['rays_o'],
        'rays_d': rays['rays_d'],
    }

    #保存结果位置
    save_path = os.path.join(opt.workspace,'my_results')
    os.makedirs(save_path,exist_ok=True)

    log(f"==> Start Test ,save results to {save_path}")
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            preds = test_step(data,model,opt)
        if opt.color_space == 'linear':
            preds = linear_to_srgb(preds)
        pred = preds[0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        print(pred.shape)
        cv2.imwrite(os.path.join(save_path, f'{opt.name}__rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

    log(f"==> Finished Test")





