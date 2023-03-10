import os
import subprocess
import argparse
from subprocess import check_output

parser = argparse.ArgumentParser()
parser.add_argument('--blender_path',type=str,
    help="Path to blender")

def run_rendering(datasetname, datasetsize, fmin=1200, fmax=2000, obj_min=0.1, obj_max=0.1, occluders=1, start_idx=0):
    args = parser.parse_args()
    blenderpath = args.blender_path
    outdir_im = os.path.join(datasetname, 'images')
    outdir_scene = os.path.join(datasetname, 'scenes')
    min_dist = 3
    max_dist = 5

    base_command = '{} --background --python render_synthetic_dataset.py -- --output_image_dir {} --num_images {} --output_scene_dir {} --start_idx {} --use_gpu 1 --occluders {} --focal_length_min {} --focal_length_max {} --min_size {} --max_size {} --min_dist {} --max_dist {}'
    run_length = 100
    timeout = 600
    for i in range(start_idx//run_length, datasetsize//run_length+1):
        start_idx = i*run_length
        end_idx = min(run_length, datasetsize-i*run_length)
        command = base_command.format(blenderpath, outdir_im, end_idx, outdir_scene, start_idx, occluders, float(fmin), float(fmax), float(obj_min), float(obj_max), float(min_dist), float(max_dist))
        print(command)
        while True:
            try:
                r = check_output(command.split(), timeout=timeout)
            except subprocess.TimeoutExpired:
                print('timeout')
                continue
            break



def main():
    obj_size = 0.1
    run_rendering('default', int(1e4), obj_min=obj_size, obj_max=obj_size, occluders=1)
    run_rendering('no_occludsion', int(1e4), obj_min=obj_size, obj_max=obj_size, occluders=0)
    run_rendering('no_occludsion_scale_variation', int(1e4), obj_min=0.8*obj_size, obj_max=1.2*obj_size, occluders=0)
    run_rendering('scale_variation', int(1e4), obj_min=0.8*obj_size, obj_max=1.2*obj_size, occluders=1)
    run_rendering('large', int(2e5), obj_min=obj_size, obj_max=obj_size, occluders=1, start_idx = 27200)

if __name__ == '__main__':
    main()
