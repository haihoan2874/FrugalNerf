import os
import argparse
import torch
import imageio
import numpy as np
from tqdm import tqdm

from dataLoader import dataset_dict
from dataLoader.ray_utils import get_rays, ndc_rays_blender
from models.tensoRF import TensorVMSplit
from utils import visualize_depth_numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--dataset_name', default='llff')
    parser.add_argument('--downsample', type=float, default=8.0)
    parser.add_argument('--num_views', type=int, default=60)
    parser.add_argument('--N_samples', type=int, default=64)
    parser.add_argument('--chunk', type=int, default=4096)
    parser.add_argument('--ndc', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    # instantiate model (assume TensorVMSplit or compatible)
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)
    tensorf.eval()

    dataset = dataset_dict[args.dataset_name]
    # build test dataset (this will load images at downsample resolution)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample, is_stack=True)
    ndc_ray = bool(args.ndc)

    # get render path and subsample to requested number of views
    c2ws = test_dataset.render_path
    if len(c2ws) > args.num_views:
        indices = np.linspace(0, len(c2ws) - 1, args.num_views).astype(int)
        c2ws = c2ws[indices]

    savePath = os.path.join(args.exp, 'imgs_path_all')
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(os.path.join(savePath, 'rgbd'), exist_ok=True)

    W, H = test_dataset.img_wh
    directions = test_dataset.directions
    white_bg = test_dataset.white_bg

    rgb_frames = []
    depth_frames = []

    for idx, c2w in enumerate(tqdm(c2ws, desc='rendering')):
        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(directions, c2w)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (H*W, 6)

        # render in chunks
        N_rays = rays.shape[0]
        rgb_all = torch.zeros((N_rays, 3), dtype=torch.float32)
        depth_all = torch.zeros((N_rays,), dtype=torch.float32)

        for i in range(0, N_rays, args.chunk):
            rays_chunk = rays[i:i+args.chunk].to(device)
            with torch.no_grad():
                rgb_map, depth_map, *_ = tensorf(rays_chunk, is_train=False, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=args.N_samples, reso=1)
            rgb_all[i:i+args.chunk] = rgb_map.cpu()
            depth_all[i:i+args.chunk] = depth_map.cpu()

        rgb_img = (rgb_all.view(H, W, 3).numpy() * 255).astype('uint8')
        depth_vis, _ = visualize_depth_numpy(depth_all.view(H, W).numpy(), None)

        imageio.imwrite(os.path.join(savePath, f'fast_{idx:03d}.png'), rgb_img)
        rgbd = np.concatenate((rgb_img, depth_vis, depth_vis, depth_vis), axis=1)
        imageio.imwrite(os.path.join(savePath, 'rgbd', f'fast_{idx:03d}.png'), rgbd)

        rgb_frames.append(rgb_img)
        depth_frames.append(depth_vis)

    # write videos
    imageio.mimwrite(os.path.join(savePath, 'fast_video.mp4'), np.stack(rgb_frames), fps=30, quality=6)
    imageio.mimwrite(os.path.join(savePath, 'fast_depthvideo.mp4'), np.stack(depth_frames), fps=30, quality=6)

    print('Done. outputs at', savePath)


if __name__ == '__main__':
    main()
