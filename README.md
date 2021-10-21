# BDInvert

This repository contains the accompanying code for [GAN Inversion for Out-of-Range Images with Geometric Transformations, ICCV 2021](https://kkang831.github.io/publication/ICCV_2021_BDInvert/)
<p align="center"><img src = "./Teaser.png" height ="500" />

## Prerequisites
- Ubuntu 18.04 or higher
- CUDA 10.0 or higher
- pytorch 1.6 or higher
- python 3.7 or higher

## Installation
```shell
pip install -r requirements.txt
```

## Usage

### Train
1. Change directory into BDInvert.
```shell
cd BDInvert
```

2. Train base code encoder.
```shell
python train_basecode_encoder.py
```
- `--model_name` : You can change backbone StyleGAN model, which located at [model zoo](MODEL_ZOO.md).
- `--basecode_spatial_size` : You can change spatial resolution of basecode.

3. Find pnorm parameters.
```shell
python pca_p_space.py
```

### Download pretrained base code encoder.
Download and unzip  under `BDInvert/pretrained_models/`.

| Encoder Pretrained Models                   | Basc Code Spatial Size |
| :--                                         | :--    |
| [StyleGAN2 pretrained on FFHQ 1024, 16x16](https://drive.google.com/file/d/1Gwi7I72vL7rdwET1Q0QnR71ZuZ0M3Jx1/view?usp=sharing)    | 16x16


### Test
* Default test setting use StyleGAN2 pretrained on FFHQ1024 and use basecode spatial size as 16x16.
1. Change directory into BDInvert.
```shell
cd BDInvert
```

2. Make image list.
```shell
python make_list.py --image_folder ./test_img
```

3. Embed images into StyleGAN's latent codes.
```shell
python invert.py --encoder_pt_path {encoder_pt_path}
```
- `--image_list` : Inversion target image list generated from above step 2. Default is ./test_img/test.list
- `--weight_pnorm_term` : As recently well known, there is a trade-off between editing quality and reconstruction quality. This argument controls this trade-off.

4. Edit embedded results.
```shell
python edit.py {inversion directory}
```
- `--edit_direction` : You can change edit direction, which located at BDInvert/editings/interfacegan_directions

## Small changes
* We changed the detail code regularization method from hard clipping in P-norm+ space to L2 norm regularization, following the update of the [original paper](https://arxiv.org/pdf/2012.09036.pdf).
* Due to this change, new hyperparameter, `weight_pnorm_term`, has been added.

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## Useful Links
* [Author Homepage](https://kkang831.github.io/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

## Acknowledgments
**NOTE**
* Our implementation is heavily based on the ["GenForce Library"](https://github.com/genforce/genforce).
* Interface GAN editing vectors are from ["encoder4editing"](https://github.com/omertov/encoder4editing).
