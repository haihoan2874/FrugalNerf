#!/usr/bin/env python3
"""
Convert COLMAP TXT model (images.txt + cameras.txt) to FrugalNeRF `poses_bounds.npy` format.

Output format: N x 17 numpy array (float32).
 - First 15 values: 3x5 pose block flattened row-major (first 3 rows x 5 cols).
   - For each of the 3 rows: first 4 values are the 3x4 camera-to-world matrix rows;
     the 5th value in row 0 = image height (H), row 1 = image width (W), row 2 = focal.
 - Last 2 values: near and far bounds (defaults 0.1, 100.0)

Usage (example):
 python tools\colmap_to_poses.py --model_txt C:\data\scene\sparse\0_txt --image_dir C:\data\scene\images --out C:\data\scene\poses_bounds.npy

This script is intentionally minimal and robust for common COLMAP TXT exports.
"""
import argparse
import os
import numpy as np
import math


def parse_cameras_txt(path):
    cameras = {}  # camera_id -> dict(model, width, height, params)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # cameras.txt format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def parse_images_txt(path):
    images = []  # list of dicts: {image_id, qvec, tvec, camera_id, name}
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        # header line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        try:
            image_id = int(parts[0])
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
        except Exception:
            # If line doesn't match header, skip
            i += 1
            continue

        images.append({
            'image_id': image_id,
            'qvec': [qw, qx, qy, qz],
            'tvec': [tx, ty, tz],
            'camera_id': cam_id,
            'name': name
        })
        i += 1
        # skip point lines until next header (COLMAP interleaves points)
        # point lines contain many values; we stop when next line starts with integer header
        while i < len(lines):
            if lines[i].split()[0].isdigit() and len(lines[i].split()) >= 9:
                break
            i += 1

    return images


def qvec2rotmat(q):
    # q = [qw, qx, qy, qz]
    qw, qx, qy, qz = q
    # normalize
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ], dtype=float)
    return R


def camera_to_world_from_qt(qvec, tvec):
    R = qvec2rotmat(qvec)
    # COLMAP stores rotation as q, and t such that: X_cam = R * X_world + t
    # So world-to-camera = [R | t]; camera-to-world = [R^T | -R^T t]
    Rt = R.T
    t = np.array(tvec).reshape((3,1))
    ctw_t = -Rt.dot(t)
    M = np.eye(4, dtype=float)
    M[:3,:3] = Rt
    M[:3,3] = ctw_t[:,0]
    return M


def infer_focal_from_camera(camera_entry):
    model = camera_entry['model']
    params = camera_entry['params']
    w = camera_entry['width']
    h = camera_entry['height']
    # Common models: PINHOLE fx fy cx cy -> use fx
    if model.upper().startswith('PINHOLE') and len(params) >= 2:
        fx = params[0]
        return float(fx)
    # SIMPLE_PINHOLE has single f param
    if model.upper().startswith('SIMPLE_PINHOLE') and len(params) >= 1:
        return float(params[0])
    # Otherwise fallback to heuristic
    return 0.5 * float(min(w, h))


def build_poses_array(images, cameras, image_dir=None, near=0.1, far=100.0):
    rows = []
    for img in images:
        cam_id = img['camera_id']
        cam = cameras.get(cam_id, None)
        if cam is None:
            # fallback: try to read image size from file
            if image_dir:
                from PIL import Image
                p = os.path.join(image_dir, img['name'])
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception:
                    w, h = 256, 256
            else:
                w, h = 256, 256
            focal = 0.5 * min(w, h)
        else:
            w = cam['width']
            h = cam['height']
            focal = infer_focal_from_camera(cam)

        M = camera_to_world_from_qt(img['qvec'], img['tvec'])

        # Build 3x5 block flattened row-major (3 rows x 5 cols)
        row = []
        for r in range(3):
            for c in range(4):
                row.append(float(M[r, c]))
            # 5th column values: row0->H, row1->W, row2->focal
            if r == 0:
                row.append(float(h))
            elif r == 1:
                row.append(float(w))
            else:
                row.append(float(focal))

        # append near/far
        row.append(float(near))
        row.append(float(far))

        if len(row) != 17:
            raise RuntimeError('Internal error building pose row length=' + str(len(row)))
        rows.append(row)

    arr = np.asarray(rows, dtype=np.float32)
    return arr


def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP TXT model to poses_bounds.npy (FrugalNeRF)')
    parser.add_argument('--model_txt', required=True, help='Path to COLMAP TXT model folder (contains cameras.txt and images.txt)')
    parser.add_argument('--image_dir', required=False, help='Optional path to image directory (used to infer sizes if cameras.txt missing)')
    parser.add_argument('--out', required=True, help='Output .npy file path (poses_bounds.npy)')
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100.0)
    args = parser.parse_args()

    cameras_txt = os.path.join(args.model_txt, 'cameras.txt')
    images_txt = os.path.join(args.model_txt, 'images.txt')
    if not os.path.exists(cameras_txt) or not os.path.exists(images_txt):
        raise FileNotFoundError('cameras.txt or images.txt not found in %s' % args.model_txt)

    cameras = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)

    poses = build_poses_array(images, cameras, image_dir=args.image_dir, near=args.near, far=args.far)

    # Save
    np.save(args.out, poses)
    print('Saved poses_bounds.npy with shape', poses.shape, '->', args.out)


if __name__ == '__main__':
    main()
