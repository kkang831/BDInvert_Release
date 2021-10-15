import sys
sys.path.append('.')
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

from image_tools import preprocess, postprocess, Lanczos_resizing
from models.stylegan_basecode_encoder import encoder_simple
from pca_p_space import project_w2pN

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
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/inversion/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--job_name', type=str, default='', help='Sub directory to save the results. If not specified, the result will be saved to {save_dir}/{model_name}')
    parser.add_argument('--image_list', type=str, default='test_img/test.list', help='target image folder path')
    parser.add_argument('--encoder_pt_path', type=str, required=True, help='base code encoder path')
    parser.add_argument('--pnorm_root', type=str, default='pnorm/stylegan2_ffhq1024')

    # Settings
    parser.add_argument('--basecode_spatial_size', type=int, default=16, help='spatial resolution of basecode.')
    parser.add_argument('--encoder_cfg', type=str, default='default')

    # Hyperparameter
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_iters', type=int, default=1200)
    parser.add_argument('--weight_perceptual_term', type=float, default=10.)
    parser.add_argument('--weight_basecode_term', type=float, default=10.)
    parser.add_argument('--weight_pnorm_term', type=float, default=0.01)
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
        work_dir = os.path.join('work_dirs', 'inversion')
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


    # Load Pnorm
    p_mean_latent = np.load(f'{args.pnorm_root}/mean_latent.npy')
    p_eigen_values = np.load(f'{args.pnorm_root}/eigen_values.npy')
    p_eigen_vectors = np.load(f'{args.pnorm_root}/eigen_vectors.npy')

    p_mean_latent = torch.from_numpy(p_mean_latent).cuda()
    p_eigen_values = torch.from_numpy(p_eigen_values).cuda()
    p_eigen_vectors = torch.from_numpy(p_eigen_vectors).cuda()

    # Load perceptual network
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    lpips_fn.net.requires_grad_(False)


    # Get GAN type
    stylegan_type = parse_gan_type(generator) # stylegan or stylegan2

    # Define layers used for base code
    basecode_layer = int(np.log2(args.basecode_spatial_size) - 2) * 2
    if stylegan_type == 'stylegan2':
        basecode_layer = f'x{basecode_layer-1:02d}'
    elif stylegan_type == 'stylegan':
        basecode_layer = f'x{basecode_layer:02d}'
    print('basecode_layer : ', basecode_layer)


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
                                      cfg=args.encoder_cfg)
    basecode_encoder.load_state_dict(torch.load(args.encoder_pt_path)['encoder'])
    basecode_encoder.cuda().eval()
    basecode_encoder.requires_grad_(False)

    #####################################
    # main
    #####################################
    image_list = []
    with open(args.image_list, 'r') as f:
        for line in f:
            image_list.append(line.strip())
    image_num = len(image_list)

    # Define save directory
    os.makedirs(os.path.join(work_dir, job_name, 'target_images'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, 'invert_results'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, 'invert_basecode'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, job_name, 'invert_detailcode'), exist_ok=True)

    # Do inversion
    for batch_idx in tqdm(range(image_num)):

        # Read images
        image_path = image_list[batch_idx]
        image_basename = os.path.splitext(os.path.basename(image_path))[0]

        image = cv2.imread(image_path)
        image_target = torch.from_numpy(preprocess(image[np.newaxis, :], channel_order='BGR')).cuda() # torch_tensor, -1~1, RGB, BCHW
        image_target = Lanczos_resizing(image_target, (generator.resolution,generator.resolution))
        image_target_resized = Lanczos_resizing(image_target, (256,256))

        target = image_target.clone()
        target_resized = image_target_resized.clone()


        # Generate starting detail codes
        detailcode_starting = generator.truncation.w_avg.clone().detach()
        detailcode_starting = detailcode_starting.view(1, 1, -1)
        detailcode_starting = detailcode_starting.repeat(1, generator.num_layers, 1)
        detailcode = detailcode_starting.clone()
        detailcode.requires_grad_(True)

        # Define starting base code
        if basecode_layer is not None:
            with torch.no_grad():
                encoder_input = Lanczos_resizing(target, (encoder_input_shape[1], encoder_input_shape[2]))
                basecode_starting = basecode_encoder(encoder_input)
                basecode = basecode_starting.clone()
            basecode.requires_grad_(True)

        #
        optimizing_variable = []
        optimizing_variable.append(detailcode)
        if basecode_layer is not None:
            optimizing_variable.append(basecode)
        optimizer = torch.optim.Adam(optimizing_variable, lr=args.lr)

        for iter in tqdm(range(args.num_iters)):

            loss = 0.
            x_rec = generator.synthesis(detailcode, randomize_noise=args.randomize_noise,
                                        basecode_layer=basecode_layer, basecode=basecode)['image']

            # MSE
            mse_loss = torch.mean((x_rec-target)**2)
            loss += mse_loss

            # LPIPS
            x_rec_resized = torch.nn.functional.interpolate(x_rec, size=(256,256), mode='bicubic')
            lpips_loss = torch.mean(lpips_fn(target_resized, x_rec_resized))
            loss += lpips_loss * args.weight_perceptual_term

            # Base code regularization
            reg_basecode_loss = torch.mean((basecode-basecode_starting)**2)
            loss += reg_basecode_loss * args.weight_basecode_term

            # Detail code regularization
            if args.weight_pnorm_term:
                pprojected_detailcode = project_w2pN(detailcode[0], p_mean_latent, p_eigen_values, p_eigen_vectors)
                reg_detailcode_loss = torch.mean((pprojected_detailcode)**2)
                loss += reg_detailcode_loss * args.weight_pnorm_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save results
        with torch.no_grad():
            x_rec = generator.synthesis(detailcode, randomize_noise=args.randomize_noise,
                                        basecode_layer=basecode_layer, basecode=basecode)['image']
            rec_image = postprocess(x_rec.clone())[0]

            cv2.imwrite(os.path.join(work_dir, job_name, 'target_images', image_basename+'.png'), postprocess(image_target.clone())[0])
            cv2.imwrite(os.path.join(work_dir, job_name, 'invert_results', image_basename+'.png'), rec_image)

            basecode_save = basecode.clone().detach().cpu().numpy()
            np.save(os.path.join(work_dir, job_name, 'invert_basecode', image_basename+'.npy'), basecode_save)

            detailcode_save = detailcode.clone().detach().cpu().numpy()
            np.save(os.path.join(work_dir, job_name, 'invert_detailcode', image_basename+'.npy'), detailcode_save)

if __name__ == '__main__':
    main()
