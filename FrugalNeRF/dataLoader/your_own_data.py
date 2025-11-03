import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T


from .ray_utils import *


class YourOwnDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, frame_num=None):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.1,100.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):
        try:
            with open(os.path.join(self.root_dir, 'frugal_dataset.txt'), 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            # Set default empty meta if file not found
            self.meta = {
                'scene_id': 'default',
                'num_images': 0,
                'scene_bbox': [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                'near_far': [0.1, 100.0],
                'white_bg': True,
                'frames': []
            }
            self.scene_bbox = torch.tensor(self.meta['scene_bbox']).view(2, 3)
            self.near_far = self.meta['near_far']
            self.white_bg = self.meta['white_bg']
            # Set default camera parameters
            self.focal_x = 500
            self.focal_y = 500
            self.cx = 256
            self.cy = 192
            w = 512
            h = 384
            self.img_wh = [w, h]
            self.meta['camera_angle_x'] = 2 * np.arctan(w / (2 * self.focal_x))
            self.meta['camera_angle_y'] = 2 * np.arctan(h / (2 * self.focal_y))
            self.meta['cx'] = self.cx
            self.meta['cy'] = self.cy
            self.meta['w'] = w
            self.meta['h'] = h
            self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y], center=[self.cx, self.cy])
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            self.intrinsics = torch.tensor([[self.focal_x, 0, self.cx], [0, self.focal_y, self.cy], [0, 0, 1]]).float()
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rays_real = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_depth = []
            self.all_depths = []
            self.all_depth_weights = []
            self.all_dense_depths = []
            return

        meta = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            if line.startswith('SCENE_ID:'):
                meta['scene_id'] = line.split(':')[1]
            elif line.startswith('NUM_IMAGES:'):
                meta['num_images'] = int(line.split(':')[1])
            elif line.startswith('SCENE_BBOX:'):
                bbox = line.split(':')[1].split(',')
                meta['scene_bbox'] = [float(x) for x in bbox]
            elif line.startswith('NEAR_FAR:'):
                nf = line.split(':')[1].split(',')
                meta['near_far'] = [float(x) for x in nf]
            elif line.startswith('WHITE_BG:'):
                meta['white_bg'] = line.split(':')[1].lower() == 'true'
            elif line == 'POSES_BOUNDS:':
                i += 1
                poses = []
                while i < len(lines) and not lines[i].strip().startswith('INTRINSICS'):
                    line = lines[i].strip()
                    if line:
                        nums = [float(x) for x in line.split()]
                        pose = nums[:16]
                        near_far = nums[16:18]
                        poses.append({'pose': pose, 'near_far': near_far})
                    i += 1
                meta['poses_bounds'] = poses
                continue
            elif line == 'INTRINSICS:':
                i += 1
                intrinsics = []
                for j in range(3):
                    line = lines[i].strip()
                    nums = [float(x) for x in line.split()]
                    intrinsics.append(nums)
                    i += 1
                meta['intrinsics'] = intrinsics
                continue
            i += 1

        self.meta = meta

        intrinsics = torch.tensor(meta['intrinsics'])
        self.focal_x = intrinsics[0, 0]
        self.focal_y = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]
        w = int(2 * self.cx)
        h = int(2 * self.cy)
        self.img_wh = [w, h]
        self.meta['camera_angle_x'] = 2 * np.arctan(w / (2 * self.focal_x))
        self.meta['camera_angle_y'] = 2 * np.arctan(h / (2 * self.focal_y))
        self.meta['cx'] = self.cx
        self.meta['cy'] = self.cy
        self.meta['w'] = w
        self.meta['h'] = h

        image_dir = os.path.join(self.root_dir, 'images')
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        frames = []
        for idx, pose_data in enumerate(meta['poses_bounds']):
            pose = pose_data['pose']
            image_file = image_files[idx]
            frames.append({
                'file_path': image_file,
                'transform_matrix': pose
            })
        self.meta['frames'] = frames
        self.scene_bbox = torch.tensor(meta['scene_bbox']).view(2, 3)
        self.near_far = meta['near_far']
        self.white_bg = meta['white_bg']


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x,self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x,0,self.cx],[0,self.focal_y,self.cy],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rays_real = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.all_depths = []
        self.all_depth_weights = []
        self.all_dense_depths = []


        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']).reshape(4,4) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, 'images', frame['file_path'])
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w) RGB
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            self.all_rays_real += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


        if len(self.poses) > 0:
            self.poses = torch.stack(self.poses)
        else:
            self.poses = torch.empty(0, 4, 4)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) if self.all_rays else torch.empty(0, 6)
            self.all_rays_real = torch.cat(self.all_rays_real, 0) if self.all_rays_real else torch.empty(0, 6)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) if self.all_rgbs else torch.empty(0, 3)
            self.all_depths = torch.cat(self.all_depths, 0) if self.all_depths else torch.empty(0)
            self.all_depth_weights = torch.cat(self.all_depth_weights, 0) if self.all_depth_weights else torch.empty(0)
            self.all_dense_depths = torch.cat(self.all_dense_depths, 0) if self.all_dense_depths else torch.empty(0)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rays_real = torch.stack(self.all_rays_real, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_depths = torch.stack(self.all_depths, 0).reshape(-1,*self.img_wh[::-1], 1) if self.all_depths else torch.empty(0)
            self.all_depth_weights = torch.stack(self.all_depth_weights, 0).reshape(-1,*self.img_wh[::-1], 1) if self.all_depth_weights else torch.empty(0)
            self.all_dense_depths = torch.stack(self.all_dense_depths, 0).reshape(-1,*self.img_wh[::-1], 1) if self.all_dense_depths else torch.empty(0)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        if len(self.poses) > 0:
            self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]
        else:
            self.proj_mat = None

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
