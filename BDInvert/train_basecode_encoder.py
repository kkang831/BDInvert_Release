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

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image
from utils.visualizer import load_image

from models.stylegan_basecode_encoder import encoder_simple
from image_tools import preprocess, postprocess, Lanczos_resizing


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='')

    # stylegan
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
    parser.add_argument('--gpu_id', type=int, default=0, help='Which gpu will be used')
    parser.add_argument('--save_dir', type=str, default='',
                        help='Root directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/train_encoder/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--job_name', type=str, default='', help='Sub directory to save the results. If not specified, the result will be saved to {save_dir}/{model_name}')

    parser.add_argument('--encoder_cfg', type=str, default='default')
    parser.add_argument('--batch_num', type=int, default=16, help='Batch size. ''(default: %(default)s)')
    parser.add_argument('--epoch_num', type=int, default=10000, help='Number of epoch. ''(default: %(default)s)')

    parser.add_argument('--perceptual_weight', type=float, default=10.)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--basecode_spatial_size', type=int, default=16, help='spatial resolution of basecode. ''(default: %(default)s)')
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
        work_dir = os.path.join('work_dirs', 'train_encoder')
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

    # Load perceptual network
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    lpips_fn.net.requires_grad_(False)

    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    #
    basecode_layer = int(np.log2(args.basecode_spatial_size) - 2) * 2
    if stylegan_type == 'stylegan2':
        basecode_layer = f'x{basecode_layer-1:02d}'
    elif stylegan_type == 'stylegan':
        basecode_layer = f'x{basecode_layer:02d}'
    print('basecode_layer : ', basecode_layer)

    #
    with torch.no_grad():
        z = torch.randn(1, generator.z_space_dim).cuda()
        w = generator.mapping(z, label=None)['w']
        wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
        basecode = generator.synthesis(wp, randomize_noise=args.randomize_noise)[basecode_layer]

    encoder_input_shape = [3, basecode.shape[2]*8, basecode.shape[3]*8]
    encoder_output_shape = basecode.shape[1:]
    print(f'Encoder input shape  = {encoder_input_shape}')
    print(f'Encoder output shape = {encoder_output_shape}')


    # Define Encoder
    basecode_encoder = encoder_simple(encoder_input_shape=encoder_input_shape,
                                     encoder_output_shape=encoder_output_shape,
                                     cfg=args.encoder_cfg).cuda()
    basecode_encoder.requires_grad_(True)
    basecode_encoder.train()

    # Define optimizer && scheduler
    print(f'Initial learning rate: {args.initial_lr}')
    optimizer = torch.optim.Adam(basecode_encoder.parameters(), lr=args.initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.1)

    # #####################################
    # # training
    # #####################################
    E_losses = []
    E_mse_losses = []
    E_lpips_losses = []

    for batch_idx in tqdm(range(args.epoch_num)):
        #######################################
        # make sample pair
        with torch.no_grad():
            z = torch.randn(args.batch_num, generator.z_space_dim).cuda()
            w = generator.mapping(z, label=None)['w']
            wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
            image = generator.synthesis(wp, randomize_noise=args.randomize_noise)['image']
            encoder_input = Lanczos_resizing(image, (encoder_input_shape[1], encoder_input_shape[2]))

        #######################################
        # training for encoder
        basecode_encoder.train()
        basecode = basecode_encoder(encoder_input)

        E_loss = 0.
        x_rec = generator.synthesis(wp, randomize_noise=args.randomize_noise,
                                    basecode_layer=basecode_layer, basecode=basecode)['image']

        # Loss MSE
        mse_loss = torch.mean((x_rec-image)**2)
        E_mse_losses.append(mse_loss.item())
        E_loss += mse_loss

        # Loss perceptual
        lpips_x_rec = torch.nn.functional.interpolate(x_rec, size=(256,256), mode='bicubic')
        lpips_gt = Lanczos_resizing(image, (256,256))
        lpips_loss = torch.mean(lpips_fn(lpips_x_rec, lpips_gt))
        E_lpips_losses.append(lpips_loss.item())
        E_loss += lpips_loss * args.perceptual_weight
        E_losses.append(E_loss.item())

        optimizer.zero_grad()
        E_loss.backward()
        optimizer.step()
        scheduler.step()

        #######################################
        # validation
        if batch_idx % 1000 == 0 or batch_idx == args.epoch_num-1:
            basecode_encoder.eval()
            os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results'), exist_ok=True)
            os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'rec_results'), exist_ok=True)
            os.makedirs(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'ground_truth'), exist_ok=True)

            plt.clf()
            plt.plot(E_losses, color='black', label=f'full_losses')
            plt.plot(E_mse_losses, color='red', label=f'mse_losses')
            plt.plot(E_lpips_losses, color='blue', label=f'lpips_losses')
            plt.legend()
            plt.savefig(os.path.join(work_dir, job_name, f'{batch_idx}_results', f'{batch_idx}_E_loss_graph.png'))

            with torch.no_grad():
                z = torch.randn(args.batch_num, generator.z_space_dim).cuda()
                w = generator.mapping(z, label=None)['w']
                wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
                image = generator.synthesis(wp, randomize_noise=args.randomize_noise)['image']
                save_image = postprocess(image.clone())
                for idx in range(args.batch_num):
                    cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'ground_truth', f'gt_{idx}.png'), save_image[idx])

                encoder_input = Lanczos_resizing(image, (encoder_input_shape[1], encoder_input_shape[2]))
                basecode = basecode_encoder(encoder_input)
                x_rec = generator.synthesis(wp, randomize_noise=args.randomize_noise,
                                    basecode_layer=basecode_layer, basecode=basecode)['image']
                save_image = postprocess(x_rec.clone())
                for idx in range(args.batch_num):
                    cv2.imwrite(os.path.join(work_dir, job_name, f'{batch_idx}_results', 'rec_results', f'rec_{idx}.png'), save_image[idx])

            checkpoint = {'encoder':basecode_encoder.state_dict(),
                          'optimizer':optimizer.state_dict(),
                          'scheduler':scheduler.state_dict(),
                          'epoch':batch_idx}

            torch.save(checkpoint, os.path.join(work_dir, job_name, f'encoder_{batch_idx}.pth'))
            torch.save(checkpoint, os.path.join(work_dir, job_name, f'encoder_final.pth'))

if __name__ == '__main__':
    main()
