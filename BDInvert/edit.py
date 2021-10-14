# -*- coding:utf-8 -*-

import sys
sys.path.append('..')

import os, inspect, shutil, json
import argparse
import subprocess
import csv
import cv2
import numpy as np
import random
import lpips
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from glob import glob

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image
from utils.visualizer import load_image

from image_tools import preprocess, postprocess, Lanczos_resizing


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # StyleGAN
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--model_name', type=str, default='stylegan2_ffhq1024',
                        help='Name to the pre-trained model.')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')

    # IO
    parser.add_argument('inversion_dir', type=str, default='',
    parser.add_argument('--gpu_id', type=int, default=0)
                        help='Latent head dir directory generated from invert.py')
    parser.add_argument('--edit_direction', type=str, default='./editings/interfacegan_directions/age.pt')
    parser.add_argument('--start_distance', type=float, default=-2.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=2.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')

    # Settings
    parser.add_argument('--use_FW_space', type=bool_parser, default=True)
    parser.add_argument('--basecode_spatial_size', type=int, default=16, help='spatial resolution of basecode.')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Set random seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Parse model configuration.
    if args.model_name not in MODEL_ZOO:
        raise SystemExit(f'Model `{args.model_name}` is not registered in 'f'`models/model_zoo.py`!')
    model_config = MODEL_ZOO[args.model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.
    # Build generation and get synthesis kwargs.
    print(f'Building generator for model `{args.model_name}` ...')
    generator = build_generator(**model_config)
    synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                            trunc_layers=args.trunc_layers,
                            randomize_noise=args.randomize_noise)
    print(f'Finish building generator.')

    # Load StyleGAN
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
    print(f'Loading StyleGAN checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    generator.requires_grad_(False)
    print(f'Finish loading StyleGAN checkpoint.')

    # Load edit direction
    direction = torch.load(args.edit_direction).cuda() # 1, 512
    def edit_code(codes, direction, weight=3):
            return codes + weight * direction

    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    # Define layers used for base code
    basecode_layer = int(np.log2(args.basecode_spatial_size) - 2) * 2
    if stylegan_type == 'stylegan2':
        basecode_layer = f'x{basecode_layer-1:02d}'
    elif stylegan_type == 'stylegan':
        basecode_layer = f'x{basecode_layer:02d}'
    print('basecode_layer : ', basecode_layer)

    # Prepare codes.
    detailcodes = np.empty((0, generator.num_layers, generator.w_space_dim))
    if args.use_FW_space:
        basecodes = np.empty((0, 512, args.basecode_spatial_size, args.basecode_spatial_size))

    inversion_dir = args.inversion_dir
    print('inversion_dir : ', inversion_dir)
    print(glob(f'{inversion_dir}/invert_results'))

    image_list = []
    for filename in glob(f'{inversion_dir}/invert_results/*.png'):
        print('file_name : ', filename)
        image_basename = os.path.splitext(os.path.basename(filename))[0]
        image_list.append(image_basename)
        detailcode = np.load(f'{inversion_dir}/invert_detailcode/{image_basename}.npy')
        detailcodes = np.concatenate([detailcodes, detailcode], axis=0)

        if args.use_FW_space:
            basecode = np.load(f'{inversion_dir}/invert_basecode/{image_basename}.npy')
            basecodes = np.concatenate([basecodes, basecode], axis=0)

    distances = np.linspace(args.start_distance,args.end_distance, args.step)

    for sam_id in tqdm(range(len(image_list)), leave=False):
        detailcode = detailcodes[sam_id:sam_id + 1]
        if args.use_FW_space:
            basecode = basecodes[sam_id:sam_id+1]
            basecode = torch.from_numpy(basecode).type(torch.FloatTensor).cuda()

        os.makedirs(os.path.join(inversion_dir, f'edited'), exist_ok=True)

        for col_id, d in enumerate(distances, start=1):
            temp_code = torch.from_numpy(detailcode.copy()).type(torch.FloatTensor).cuda()
            temp_code = edit_code(temp_code, direction, weight=d)

            if args.use_FW_space:
                image = generator.synthesis(temp_code, randomize_noise=args.randomize_noise,
                                            basecode_layer=args.basecode_layer, basecode=basecode)['image']
            else:
                image = generator.synthesis(temp_code, randomize_noise=args.randomize_noise)['image']

            image = postprocess(image)[0]
            cv2.imwrite(os.path.join(inversion_dir, f'edited', f'{image_list[sam_id]}_{col_id}.png'), image)

if __name__ == '__main__':
    main()
