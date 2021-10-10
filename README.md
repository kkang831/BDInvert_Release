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
1. Change directory into BDInvert.
```shell
cd BDInvert
```

2. Make image list.
```shell
python make_list.py {image folder}
```

3. Embed images into StyleGAN's latent codes.
```shell
python invert.py {image_list path}
```

4. Edit embedded results.
```shell
python edit.py {inversion directory}
```

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## Useful Links
* [Author Homepage](https://kkang831.github.io/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

## Related projects
**NOTE** : Our implementation is heavily based on the ["GenForce Library"](https://github.com/genforce/genforce)
