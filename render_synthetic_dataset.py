# modified render_images.py from CLEVR implementation

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import time
random.seed(time.time())

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector, Matrix
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
BASE_SCENE_BLENDFILE = 'data/base_scene.blend'
PROPERTIES_FILE = 'data/properties.json'
SHAPE_DIR='data/shapes'
MATERIALS_DIR = 'data/materials'

# Settings for scene
parser.add_argument('--min_dist', default=3, type=float,
    help="The minimum allowed depth, meters")
parser.add_argument('--max_dist', default=5, type=float,
    help="The maximum allowed depth, meters")
parser.add_argument('--min_size', default=1, type=float,
    help="minimum size of main object")
parser.add_argument('--max_size', default=1, type=float,
    help="maximum size of main object")
parser.add_argument('--occluders', default=1, type=int,
    help="whether to include occluders or not")
parser.add_argument('--min_size_occluder', default=0.5, type=float,
    help="minimum size of other object")
parser.add_argument('--max_size_occluder', default=2, type=float,
    help="maximum size of other object")
parser.add_argument('--focal_length_min', default=80, type=float)
parser.add_argument('--focal_length_max', default=120, type=float)

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=224, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=224, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=5.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=5.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=5.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):
    img_template = 'sample%06d.png'
    scene_template = 'sample%06d.json'
    img_template = os.path.join(args.output_image_dir, img_template)
    scene_template = os.path.join(args.output_scene_dir, scene_template)

    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)

    all_scene_paths = []
    for i in range(args.num_images):
        output_index = (i + args.start_idx)
        img_path = img_template % output_index
        scene_path = scene_template % output_index
        all_scene_paths.append(scene_path)
        render_scene(args,
          output_index=output_index,
          output_image=img_path,
          output_scene=scene_path,
        )
    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))


def render_scene(args,
    output_index=0,
    output_image='render.png',
    output_scene='render_json'):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=BASE_SCENE_BLENDFILE)

    # Load materials
    utils.load_materials(MATERIALS_DIR)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    # render_args.tile_x = args.render_tile_size
    # render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'visibility': -1,
        'focal_length': -1,
    }

    camera = bpy.data.objects['Camera']
    # Save all six axis-aligned directions in the scene struct

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    pos_main, focal_length, visibility_ratio = add_random_objects(scene_struct, args, camera)

    scene_struct['pos_main'] = [pos_main[0], pos_main[1], pos_main[2]]
    scene_struct['focal_length'] = focal_length
    scene_struct['visibility'] = visibility_ratio
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)


def add_random_objects(scene_struct, args, camera):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(PROPERTIES_FILE, 'r') as f:
        properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())
    focal_length = sample_uniform(args.focal_length_min, args.focal_length_max)
    bpy.data.cameras.values()[0].sensor_width = args.width/focal_length*bpy.data.cameras.values()[0].lens
    bpy.data.cameras.values()[0].sensor_height = args.width/focal_length*bpy.data.cameras.values()[0].lens

    camera = bpy.data.objects['Camera']
    cam_orientation = camera.matrix_world
   
    # sample position for main object
    main_z = sample_uniform(args.min_dist, args.max_dist)
    rx = (random.random()-0.5)
    ry = (random.random()-0.5)
    main_x = rx * main_z * args.width / 2 / focal_length
    main_y = ry * main_z * args.height / 2 / focal_length
    pos_camera = Vector((main_x, main_y, -main_z, 1.0))
    pos_main = cam_orientation @ pos_camera

    # add object
    scale_main = sample_uniform(args.min_size, args.max_size)
    objtype = 'SmoothCylinder'
    objname = 'obj_main'
    _, rgba = random.choice(list(color_name_to_rgba.items()))
    mat_name, mat_name_out = random.choice(material_mapping)
    physical_objects = []
    physical_objects.append(add_object(objname, scale_main, objtype, pos_main, rgba, mat_name))
    gt_pos = (main_x, main_y, main_z)
    num_pixels_full = check_visibility('obj_main', physical_objects)
    if args.occluders:
        for i, objtype in zip(range(2), ['SmoothCube_v2', 'Sphere']):
            scale = sample_uniform(args.min_size, args.max_size)
            # sample position for occluding object
            z = sample_uniform(args.min_dist, args.max_dist)
            x = (random.random()-0.5)*2 * main_z * args.width / 2 / focal_length
            y = (random.random()-0.5)*2 * main_z * args.height / 2 / focal_length
            pos_camera = Vector((x, y, -z, 1.0))
            pos_obj = cam_orientation @ pos_camera
            _, rgba = random.choice(list(color_name_to_rgba.items()))
            mat_name, mat_name_out = random.choice(material_mapping)
            objname = 'obj_{}'.format(i)
            physical_objects.append(add_object(objname, scale, objtype, pos_obj, rgba, mat_name))
    num_pixels_occ = check_visibility('obj_main', physical_objects)
    if num_pixels_full == 0:
        visibility_ratio = 0.0
    else:
        visibility_ratio = num_pixels_occ / num_pixels_full
    return gt_pos, focal_length, visibility_ratio

def add_object(objname, scale, objtype, position, color, mat_name):
    filename = os.path.join(SHAPE_DIR, '{}.blend', 'Object/{}').format(objtype, objtype)
    bpy.ops.wm.append(filename=filename)
    active = bpy.data.objects.get(objtype)
    active.name = objname
    r = (2*random.random()-1)*3.1415
    az = (2*random.random()-1)*3.1415
    p = (2*random.random()-1)*3.1415/2
    active.rotation_euler[0] = p
    active.rotation_euler[1] = r
    active.rotation_euler[2] = az
    active.scale = (scale, scale, scale)
    active.location[0] = position[0]
    active.location[1] = position[1]
    active.location[2] = position[2]
    utils.add_material(active, mat_name, Color=color)
    return active

def check_visibility(target_name, blender_objects):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((int(p[i]>0.5), int(p[i+1]>0.5), int(p[i+2]>0.5), int(p[i+3]>0.5))
                        for i in range(0, len(p), 4))
  os.remove(path)
  target_color = object_colors[target_name]
  target_color = [int(v) for v in target_color]
  for color, count in color_count.most_common():
      if target_color[0] == color[0] and target_color[1] == color[1] and target_color[2] == color[2]:
         return count
  return 0


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_EEVEE'

  # Add random shadeless materials to all objects
  object_colors = {}
  old_materials = []
  mat_new = []
  for i, obj in enumerate(blender_objects):
      old_materials.append(obj.data.materials[0])
      mat_name = 'Material_new_%d' % i
      mat = bpy.data.materials.new(mat_name)
      while True:
          r, g, b = tuple([float(random.random() > 0.5) for _ in range(3)])
          if (r,g,b) != (0,0,0) and (r, g, b) not in object_colors: break
      object_colors[obj.name] = (r,g,b)
      mat.use_nodes = True
      mat.node_tree.nodes.clear()
      node_emission = mat.node_tree.nodes.new(type="ShaderNodeEmission")
      node_emission.inputs[0].default_value = (r,g,b,1)
      node_emission.inputs[1].default_value = 1.0
      node_output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
      mat.diffuse_color = [r,b,g,1]
      mat.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])
      obj.data.materials[0] = mat
      mat_new.append(mat)

  # Render the scene
  bpy.ops.render.render(write_still=True)

  for mat, obj in zip(old_materials, blender_objects):
      obj.data.materials[0] = mat

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine

  return object_colors


def sample_uniform(minv, maxv):
    return random.random()*(maxv-minv)+minv


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')
