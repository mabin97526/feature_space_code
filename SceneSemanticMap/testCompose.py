import json

import torch
import argparse
from nerf.provider import NeRFDataset, nerf_matrix_to_ngp
from nerf.gui import NeRFGUI
from tensoRF.utils import *
from scipy.spatial.transform import Rotation as Rot
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type = str,help="dataset")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")

    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--lr0', type=float, default=2e-2, help="initial learning rate for embeddings")
    parser.add_argument('--lr1', type=float, default=1e-3, help="initial learning rate for networks")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--l1_reg_weight', type=float, default=1e-5)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--resolution0', type=int, default=128)
    parser.add_argument('--resolution1', type=int, default=300)
    parser.add_argument("--upsample_model_steps", type=int, action="append", default=[2000, 3000, 4000, 5500, 7000])
    parser.add_argument('--color_space', type=str, default='linear', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    parser.add_argument('--name',type=str,default='composeNeRF')
    opt = parser.parse_args()
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."
    print(opt)
    seed_everything(opt.seed)

    assert opt.cuda_ray, 'CCNeRF only supports CUDA raymarching mode'

    from tensoRF.network_cc import NeRFNetwork as CCNeRF

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CCNeRF(
        rank_vec_density=[1],
        rank_mat_density=[1],
        rank_vec=[1],
        rank_mat=[1],
        resolution=[1] * 3,  # fake resolution
        bound=opt.bound,  # a large bound is needed
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    ).to(device)

    print(model)
    def load_model(path):
        checkpoint_dict = torch.load(path, map_location=device)
        model = CCNeRF(
            rank_vec_density=checkpoint_dict['rank_vec_density'],
            rank_mat_density=checkpoint_dict['rank_mat_density'],
            rank_vec=checkpoint_dict['rank_vec'],
            rank_mat=checkpoint_dict['rank_mat'],
            resolution=checkpoint_dict['resolution'],
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
        ).to(device)
        model.load_state_dict(checkpoint_dict['model'],strict=False)
        return model


    cube = load_model('./recurrent_ppo/composemodel/cube_feature.pth')
    limo = load_model('trial_limo_new/checkpoints/64_16-64_64.pth')
    ball = load_model('./recurrent_ppo/composemodel/ball_feature.pth')
    stair = load_model('./trial_stair_new/checkpoints/64_16-64_64.pth')
    capsule = load_model('./trial_capsule_new/checkpoints/64_16-64_64.pth')
    model.compose(ball,s=0.5,t=np.array([0.3,-0.5,-0.2]),R=Rot.from_euler('zyx', [0,0,0], degrees=True).as_matrix())
    #print(ball)

    opt.ckpt = 'scratch'

    #print(model)
    transform_path = os.path.join(opt.path,'test.json')
    with open(transform_path,'r') as f:
        transform = json.load(f)
    if 'h' in transform and 'w' in transform:
        H = int(transform['h'])
        W = int(transform['w'])

    frames = transform["frames"]

    poses = []
    images = []
    for f in tqdm.tqdm(frames ,desc=f'Loading test data'):

        pose = np.array(f['transform_matrix'],dtype=np.float32)


        pose = nerf_matrix_to_ngp(pose,scale = opt.scale,offset=opt.offset)

        poses.append(pose)

    poses = torch.from_numpy(np.stack(poses,axis=0)).to(device)
    radius = poses[:, :3, 3].norm(dim=-1).mean(0).item()
    fl_x = transform['fl_x']
    fl_y = transform['fl_y']
    cx = transform['cx']
    cy = transform['cy']
    intrinsics = np.array([fl_x,fl_y,cx,cy])

    rays = get_rays(poses,intrinsics,H,W,-1)
    data = {
        'H': H,
        'W': W,
        'rays_o': rays['rays_o'],
        'rays_d': rays['rays_d'],
    }



    save_path = os.path.join(opt.workspace,'Myresult')
    os.makedirs(save_path,exist_ok=True)
    print("Start Test, save results to {}".format(save_path))
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            preds = test_step(data,model,opt)
            #data,model,opt,bg_color=None,perturb=False,device='cuda'
        pred = preds[0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path,f'{opt.name}__rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    print("====>> Finished Test")
    print("====>> Second Test")
    t1 = time.time()
    model = CCNeRF(
        rank_vec_density=[1],
        rank_mat_density=[1],
        rank_vec=[1],
        rank_mat=[1],
        resolution=[1] * 3,  # fake resolution
        bound=opt.bound,  # a large bound is needed
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    ).to(device)


    #model.compose(cube, s=1, t=np.array([0, -1, -1]))
    #model.compose(ball, s=1, t=np.array([0, 0, 1]))

    #model.compose(limo,s=0.5,t=np.array([0,0,1]))
    model.compose(ball, s=0.5, t=np.array([0, 1, 0]))
    model.compose(ball, s=0.5, t=np.array([0, 0, -0.5]))
    #model.compose(stair, s=1, t=np.array([0, 0, 0.5]),R=Rot.from_euler('zyx', [0,-75,0], degrees=True).as_matrix())
    model.compose(cube, s=0.5, t=np.array([0.5, 0, 0]),R=Rot.from_euler('zyx', [0,-75,0], degrees=True).as_matrix())
    model.compose(cube, s=0.7, t=np.array([-0.3, 0, 0]))
    model.compose(ball, s=0.5, t=np.array([0, -1, 0.8]))
    model.compose(ball, s=0.5, t=np.array([-0.3, -0.8, 0]))
    #model.compose(stair, s=1, t=np.array([0, 0, 0]))

    #print(model)
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            preds = test_step(data,model,opt)
        pred = preds[0].detach().cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path,f'{opt.name}__2rgb.png'),cv2.cvtColor(pred,cv2.COLOR_RGB2BGR))
    t1 = time.time() -t1
    print(t1)
    print("====>> Finished2Test")
