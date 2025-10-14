"""Small helper: create MP4 from imgs_vis, render a PLY preview, and save sample histogram/RGBA.

Usage (PowerShell):
    python tools\make_outputs.py --exp log/fern_4v_light

This script will try to use imageio-ffmpeg / imageio to write mp4, and trimesh + pyrender to render a PLY preview.
It falls back to simpler behavior if optional libs are missing.
"""
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True, help='experiment folder under log, e.g. log/fern_4v_light')
args = parser.parse_args()

exp_path = Path(args.exp)
if not exp_path.exists():
    print(f'Error: {exp_path} does not exist')
    sys.exit(2)

imgs_vis = exp_path / 'imgs_vis'
imgs_rgba = exp_path / 'imgs_rgba'
ply_file = exp_path / f'{exp_path.name}.ply'

print('Found:')
print(' - imgs_vis =', imgs_vis)
print(' - imgs_rgba =', imgs_rgba)
print(' - ply =', ply_file)

# 1) Make MP4 from imgs_vis
mp4_out = exp_path / f'{exp_path.name}_vis.mp4'
pngs = sorted(imgs_vis.glob('*.png')) if imgs_vis.exists() else []
if not pngs:
    print('No PNGs found under imgs_vis, skipping mp4 creation')
else:
    try:
        import imageio
        writer = imageio.get_writer(str(mp4_out), fps=24)
        for p in pngs:
            img = imageio.imread(str(p))
            writer.append_data(img)
        writer.close()
        print('Wrote MP4 ->', mp4_out)
    except Exception as e:
        print('imageio write failed:', e)
        print('Try installing imageio and imageio-ffmpeg: pip install imageio imageio-ffmpeg')

# 2) Save a PLY preview (render a simple orthographic snapshot) using trimesh + pyrender
ply_preview = exp_path / 'ply_preview.png'
try:
    import trimesh
    import pyrender
    mesh = None
    if ply_file.exists():
        mesh = trimesh.load(str(ply_file))
    if mesh is None:
        raise FileNotFoundError('PLY not found or could not be loaded')

    scene = pyrender.Scene(bg_color=[255,255,255,0], ambient_light=[0.3,0.3,0.3])
    mesh_tr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_tr)
    # add camera
    camera = pyrender.PerspectiveCamera(yfov=1.047)
    cam_node = scene.add(camera, pose=[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]])
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=3.0)
    scene.add(light, pose=[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]])
    r = pyrender.OffscreenRenderer(800,600)
    color, depth = r.render(scene)
    from PIL import Image
    Image.fromarray(color).save(str(ply_preview))
    r.delete()
    print('Wrote PLY preview ->', ply_preview)
except Exception as e:
    print('PLY preview skipped (missing trimesh/pyrender or failed):', e)
    print('To enable: pip install trimesh pyrender pyglet')

# 3) Copy a sample RGBA image if exists
sample_rgba = exp_path / 'sample_rgba.png'
try:
    if imgs_rgba.exists():
        first_rgba = next(imgs_rgba.glob('*.png'), None)
        if first_rgba:
            from shutil import copyfile
            copyfile(first_rgba, sample_rgba)
            print('Copied sample RGBA ->', sample_rgba)
        else:
            print('No RGBA PNG found under imgs_rgba')
    else:
        print('imgs_rgba not present')
except Exception as e:
    print('Failed copying RGBA sample:', e)

# 4) Find a histogram png under imgs_vis/histograms or imgs_test_all/histograms and copy one
hist_src = None
for candidate in (imgs_vis / 'histograms', exp_path / 'imgs_test_all' / 'histograms'):
    if candidate.exists():
        hist = next(candidate.glob('*.png'), None)
        if hist:
            hist_src = hist
            break
if hist_src:
    from shutil import copyfile
    hist_out = exp_path / 'sample_histogram.png'
    copyfile(hist_src, hist_out)
    print('Copied histogram ->', hist_out)
else:
    print('No histogram PNG found')

print('All done')
