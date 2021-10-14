import sys
sys.path.append('.')
sys.path.append('..')

import os, inspect, shutil, json
import argparse
import subprocess
import csv
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F


from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser

def project_w2pN(w, mean_latents, eigen_values, eigen_vectors):
    p = F.leaky_relu(w, negative_slope=5.)
    p = p - mean_latents
    return torch.matmul(p, eigen_vectors) / torch.sqrt(eigen_values.T).expand(p.shape[0],-1)

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # StyleGAN
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--model_name', type=str,
                        help='Name to the pre-trained model.', default='stylegan2_ffhq1024')
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
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='',
                        help='Root directory to save the results. If not specified, '
                             'the results will be saved to `pnorm/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--job_name', type=str, default='', help='Sub directory to save the results. If not specified, the result will be saved to {save_dir}/{model_name}')

    # Settings
    parser.add_argument('--total_samples_num', type=int, default=1000000, help='Number of random samples')
    parser.add_argument('--batch_num', type=int, default=10000)
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

    # Get work directory and job name.
    if args.save_dir:
        work_dir = args.save_dir
    else:
        work_dir = os.path.join('pnorm')
    os.makedirs(work_dir, exist_ok=True)
    job_name = args.job_name
    if job_name == '':
        job_name = f'{args.model_name}'
    os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

    # Save current file and arguments
    current_file_path = inspect.getfile(inspect.currentframe())
    current_file_name = os.path.basename(current_file_path)
    shutil.copyfile(current_file_path, os.path.join(work_dir, job_name, current_file_name))
    with open(os.path.join(work_dir, job_name, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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

    total_samples_num = args.total_samples_num
    batch_num = args.batch_num

    latents = np.empty((0, generator.z_space_dim), dtype=np.float32)
    for _ in range(total_samples_num//batch_num):
        z = torch.randn(batch_num, generator.z_space_dim).cuda()
        w = generator.mapping(z)['w']
        w = F.leaky_relu(w, negative_slope=5.)
        latents = np.concatenate((latents, w.cpu().numpy()), axis=0)

    mean_latents = latents.mean(axis=0)
    latents = latents - mean_latents
    cov = np.cov(latents.T)

    # PCA
    # A: eigen values
    # C: eigen vectors
    A, C = np.linalg.eig(cov)

    np.save(os.path.join(work_dir, job_name, f'mean_latent.npy'), mean_latents)
    np.save(os.path.join(work_dir, job_name, f'eigen_values.npy'), A.astype(np.float32))
    np.save(os.path.join(work_dir, job_name, f'eigen_vectors.npy'), C.astype(np.float32))

    #####################################
    # Testbed
    #####################################
    latents = torch.empty((0, generator.z_space_dim), dtype=torch.float32).cuda()
    for _ in range(total_samples_num//batch_num):
        z = torch.randn(batch_num, generator.z_space_dim).cuda()
        w = generator.mapping(z)['w']
        latents = torch.cat((latents, w), axis=0)

    mean_latents = torch.from_numpy(mean_latents).cuda()
    eigen_values = torch.from_numpy(A.astype(np.float32)).cuda()
    eigen_vectors = torch.from_numpy(C.astype(np.float32)).cuda()

    projected = project_w2pN(latents, mean_latents, eigen_values, eigen_vectors)
    print('projected.shape : ', projected.shape)
    print('projected.mean : ', projected.mean(axis=0))
    print('projected.std : ', projected.std(axis=0))

if __name__ == '__main__':
    main()
