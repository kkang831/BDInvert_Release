import numpy as np
import torch
import PIL
from PIL import Image


def preprocess(images, channel_order='RGB'):
    """Preprocesses the input images if needed.
    This function assumes the input numpy array is with shape [batch_size,
    height, width, channel]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, channel, height, width].
    NOTE: The channel order of input images is always assumed as `RGB`.
    Args:
      images: The raw inputs with dtype `numpy.uint8` and range [0, 255].
    Returns:
      The preprocessed images with dtype `numpy.float32` and range
        [-1, 1].
    """
    # input : numpy, np.uint8, 0~255, RGB, BHWC
    # output : numpy, np.float32, -1~1, RGB, BCHW

    image_channels = 3
    max_val = 1.0
    min_val = -1.0

    if image_channels == 3 and channel_order == 'BGR':
      images = images[:, :, :, ::-1]
    images = images / 255.0 * (max_val - min_val) + min_val
    images = images.astype(np.float32).transpose(0, 3, 1, 2)
    return images

def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    # input : tensor, -1~1, RGB, BCHW
    # output : np.uint8, 0~255, BGR, BHWC

    images = images.detach().cpu().numpy()
    images = (images + 1.) * 255. / 2.
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)[:,:,:,[2,1,0]]
    return images

def Lanczos_resizing(image_target, resizing_tuple=(256,256)):
    # input : -1~1, RGB, BCHW, Tensor
    # output : -1~1, RGB, BCHW, Tensor
    image_target_resized = image_target.clone().cpu().numpy()
    image_target_resized = (image_target_resized + 1.) * 255. / 2.
    image_target_resized = np.clip(image_target_resized + 0.5, 0, 255).astype(np.uint8)

    image_target_resized = image_target_resized.transpose(0, 2, 3, 1)
    tmps = []
    for i in range(image_target_resized.shape[0]):
        tmp = image_target_resized[i]
        tmp = Image.fromarray(tmp) # PIL, 0~255, uint8, RGB, HWC
        tmp = np.array(tmp.resize(resizing_tuple, PIL.Image.LANCZOS))
        tmp = torch.from_numpy(preprocess(tmp[np.newaxis,:])).cuda()
        tmps.append(tmp)
    return torch.cat(tmps, dim=0)
