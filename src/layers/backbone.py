from typing import Tuple
from typing import Union

import numpy as np
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
from einops import rearrange
from torch import nn
from torchvision import transforms

from src.utils import helper


class IdentityEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # resolve the issue of `safetensors_rust.SafetensorError:
        # Error while deserializing header: InvalidHeaderDeserialization`
        self.register_buffer("dummy", torch.tensor([0.0]))

    def encode(self, x: torch.Tensor):
        return x

    def decode(self, x: torch.Tensor):
        return x


class UNetEncoder(UNet2DModel):

    def __init__(self, input_channels=3, *args, **kargs):
        super().__init__(*args, **kargs)
        in_channels = kargs["in_channels"]
        self.downscale_cnn = nn.Conv2d(input_channels, in_channels, kernel_size=4, stride=4)

    def forward(
        self,
        sample: torch.Tensor,
    ) -> Union[UNet2DOutput, Tuple]:
        sample = self.downscale_cnn(sample)
        sample = super().forward(sample, timestep=0, class_labels=None).sample

        return sample


class DINOEncoder(nn.Module):

    def __init__(
        self,
        model_name="dinov2_vitb14",
        requires_grad: bool = False,
        sample_size=32,
        out_channels=768,
        enable_register: bool = False,
        *args, **kargs,
    ):

        super().__init__()
        if enable_register:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name, f"{model_name}_reg")
        else:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", model_name)

        self.requires_grad = requires_grad
        self.transform = transforms.Compose(
            [
                transforms.Resize(sample_size * 14),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        helper.set_grad(self.dinov2, requires_grad)
        if requires_grad:
            self.dinov2.train()
            delattr(self.dinov2, "mask_token")
        else:
            self.dinov2.eval()

        self.sample_size = sample_size
        self.out_channels = out_channels

    def forward(self, x):
        # expect input in range [-1, 1]
        with torch.set_grad_enabled(self.requires_grad):
            x = self.transform(x * 0.5 + 0.5)
            enc_out = self.dinov2.forward_features(x)
            return rearrange(
                enc_out["x_norm_patchtokens"],
                "b (h w) c -> b c h w",
                h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
            )

    def train(self, mode: bool = True):
        if self.requires_grad:
            self.dinov2.train(mode)

        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the module."""
        state = {}
        if self.requires_grad:
            state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state
