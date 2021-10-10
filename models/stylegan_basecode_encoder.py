import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Any, cast

class encoder_simple(nn.Module):
    def __init__(self, encoder_input_shape, encoder_output_shape, cfg='default', filter_min=128, filter_max = 512, batch_norm=True, init_weights=True):
        super(encoder_simple, self).__init__()

        if len(encoder_input_shape) != 3:
            raise ValueError(f"encoder_input_shape format is not valid! current shape is {encoder_input_shape}")

        if len(encoder_output_shape) != 3:
            raise ValueError(f"encoder_output_shape format is not valid! current shape is {encoder_output_shape}")

        in_channels = encoder_input_shape[0]
        encoder_input_size = encoder_input_shape[1]

        output_channels = encoder_output_shape[0]
        encoder_output_size = encoder_output_shape[1]

        cfgs: Dict[str, List[Union[str, int]]] = {
            'default': [],
        }

        if cfg == 'default':
            block_nums = int(np.log2(encoder_input_size//encoder_output_size))
            out_channel = filter_min
            for _ in range(block_nums):
                cfgs[cfg].append(out_channel)
                cfgs[cfg].append(out_channel)
                cfgs[cfg].append(out_channel)
                cfgs[cfg].append('P')
                out_channel = min(2*out_channel, filter_max)
            cfgs[cfg].append(output_channels)
            cfgs[cfg].append(output_channels)

        cfg = cfgs[cfg]
        layers = []
        for idx, v in enumerate(cfg):
            if v == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if idx == len(cfg)-1:
                    layers += [conv2d]
                    break
                else:
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                    else:
                        layers += [conv2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                in_channels = v
        self.model = nn.Sequential(*layers)

        print(f'StyleGAN BaseCode Encoder Architecture')
        print(self.model)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        return x
