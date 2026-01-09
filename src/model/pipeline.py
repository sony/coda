import os
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import diffusers
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import \
    rescale_noise_cfg
from diffusers.utils import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from packaging import version
from safetensors.torch import load_file
from safetensors.torch import save_file
from torch import nn

from ..utils import helper

logger = logging.get_logger("diffusers")


def pad_embeddings(embeds: torch.Tensor, seq_len: int, drop_rate: float = 0.0):
    """Padding to the embeddings.

    Args:
        embeds (torch.Tensor): Input tensor of shape (batch_size, n, D)
        seq_len (int): The length of output sequence

    Returns:
        torch.Tensor: Nagative prompt embeddings of shape (batch_size, seq_len, D)
        torch.Tensor: Boolean tensor of shape (batch_size, seq_len)
    """
    masks = torch.ones_like(embeds[..., 0]) > 0
    length = embeds.shape[1]

    if drop_rate > 0:
        drop = torch.rand_like(embeds[..., 0])

    if length < seq_len:
        embeds = F.pad(embeds, (0, 0, 0, seq_len-length), mode="constant", value=0)
        masks = F.pad(masks, (0, seq_len-length), mode="constant", value=False)
        if drop_rate > 0:
            drop = F.pad(drop, (0, seq_len-length), mode="constant", value=1.0)

    embeds = embeds[:, :seq_len, :]
    masks = masks[:, :seq_len]

    if drop_rate > 0:
        masks = torch.logical_and(masks, drop > drop_rate)

    return embeds, masks


def enable_grad(parameters, grad):
    for p in parameters:
        p.requires_grad_(grad)


class DiffusionPipeline(nn.Module):

    def __init__(
        self,
        model_id: str,
        scheduler_train: str,
        scheduler_test: str,
        encoder: nn.Module,
        inference_steps: int = 50,
        guidance_scale: float = None,
        enable_xformers: bool = False,
        training_unet_cross_attention: bool = True,
        training_unet: bool = False,
        mixed_precision: str = "no",
        cfg_dropping_rate: float = 0,
        guidance_rescale: float = 0,
        trainable_keys: Tuple[str] = None,
        share_slot_init: bool = True,
        contrastive_weight: float = 0.1,
        contrastive_sampling_rate: float = 0.5,
        contrastive_warmup: int = 20000,
        *args, **kwargs,
    ):
        super().__init__()

        weight_dtype = torch.float32
        if mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        self.trainable_keys = trainable_keys

        excluded_params = set()

        self.scheduler_train = getattr(diffusers, scheduler_train).from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.scheduler_test = getattr(diffusers, scheduler_test).from_pretrained(
            model_id, subfolder="scheduler"
        )

        # ----------------------------------------------------------------------------#
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.eval()

        # remove parameters in state_dict
        for name, _ in self.vae.named_parameters("vae"):
            excluded_params.add(name)
        # ----------------------------------------------------------------------------#

        # ----------------------------------------------------------------------------#
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.unet.requires_grad_(training_unet)

        if training_unet:
            print("Finetuning UNet")
        else:
            self.unet.eval()

            # remove parameters in state_dict
            for name, param in self.unet.named_parameters("unet"):
                requires_grad = False
                for p in self.trainable_keys:
                    requires_grad |= training_unet_cross_attention and (p in name)

                if requires_grad:
                    param.requires_grad_(requires_grad)
                    print(f"Enabling grad for {name}")
                else:
                    excluded_params.add(name)

        self.unet_trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        # ----------------------------------------------------------------------------#

        if enable_xformers:
            import xformers
            xformers_version = version.parse(xformers.__version__)

            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers "
                    "to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers "
                    "for more details."
                )

            print(f"Enabling xformers for unet and vae")
            self.unet.enable_xformers_memory_efficient_attention()
            self.vae.enable_xformers_memory_efficient_attention()

        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.training_unet_cross_attention = training_unet_cross_attention
        self.training_unet = training_unet
        self.model_id = model_id
        self.cfg_dropping_rate = cfg_dropping_rate
        self.share_slot_init = share_slot_init

        self.contrastive_weight = contrastive_weight
        self.contrastive_sampling_rate = contrastive_sampling_rate
        self.contrastive_warmup = contrastive_warmup

        self.encoder = encoder
        self.excluded_params = excluded_params

    def prepare_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ):
        """Compute hidden states."""

        attention_mask = None

        if self.cfg_dropping_rate > 0:
            bs = hidden_states.shape[0]
            drop_mask = torch.rand(bs, 1, 1, device=self.device) < self.cfg_dropping_rate

            negative_embeds = self.encoder.negative_prompt(bs)
            seq_len = max(hidden_states.shape[1], negative_embeds.shape[1])

            negative_embeds, negative_masks = pad_embeddings(negative_embeds, seq_len)
            hidden_states, attention_mask = pad_embeddings(hidden_states, seq_len)

            hidden_states = torch.where(drop_mask, negative_embeds, hidden_states)
            attention_mask = torch.where(drop_mask.squeeze(-1), negative_masks, attention_mask)

        return hidden_states, attention_mask

    def prepare_negative_hidden_states(
        self,
        slots,
    ):
        """
        Compute contrative hidden_states.

        Args:
            slots (torch.Tensor): slots of shape (B, L, C).
        Return:
            torch.Tensor: Negative hidden states of shape (B, L, C).
        """
        bs, n_slots = slots.shape[:2]

        skip_mask = torch.rand(bs, n_slots, 1, device=self.device) > self.contrastive_sampling_rate
        shifted_slots = torch.cat((slots[bs//2:], slots[:bs//2]), dim=0)
        shifted_slots = torch.where(skip_mask, slots, shifted_slots)

        hidden_states = self.encoder.compute_hidden_states({"slots": shifted_slots})

        return hidden_states, None

    def compute_error(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Compute the errors.

        Args:
            noisy_latent (torch.Tensor): Noisy latent of shape (B, C, H, W).
            timesteps (torch.Tensor): Timesteps.
            hidden_states (torch.Tensor): Hidden states of shape (B, L, C).
            attention_mask (torch.Tensor): Attention mask of shape (B, L).
            target (torch.Tensor): Target of shape (B, C, H, W).
        Return:
            torch.Tensor: MSE loss of shape (B,).
        """
        model_pred = self.unet(
            noisy_latent,
            timesteps,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False
        )[0]

        errors = F.mse_loss(model_pred, target, reduction="none")
        errors = helper.sum_flat(errors) / model_pred[0].numel()

        return errors

    def forward(self, batch, *args, **kargs):
        images = batch["image"]

        n_slots = self.encoder.config.slot_n_slots
        d_slots = self.encoder.config.slot_size
        bs, device = images.shape[0], images.device

        if self.share_slot_init:
            slot_noise = torch.randn(bs//2, n_slots, d_slots, device=device)
            slot_noise = torch.cat([slot_noise, slot_noise], dim=0)
        else:
            slot_noise = torch.randn(bs, n_slots, d_slots, device=images.device)

        slot_outputs = self.encoder(images, slot_noise)
        hiddens_pos = self.encoder.compute_hidden_states(slot_outputs)

        # sample a random timestep for each image
        max_step = self.scheduler_train.config.num_train_timesteps
        timesteps = torch.randint(0, max_step, (bs,))
        timesteps = timesteps.long().to(device)

        with torch.no_grad():
            latent = self.vae.encode(images).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor

        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler_train.add_noise(latent, noise, timesteps)

        if self.scheduler_train.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_train.config.prediction_type == "v_prediction":
            target = self.scheduler_train.get_velocity(latent, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_train.config.prediction_type}")

        hiddens_pos, masks_pos = self.prepare_hidden_states(hiddens_pos)
        hiddens_neg, masks_neg = self.prepare_negative_hidden_states(slot_outputs["slots"])

        step = kargs["step"]
        weight_contrastive = self.contrastive_weight if step > self.contrastive_warmup else 0

        # -------------------------- negative loss ------------------------------------------------
        if weight_contrastive > 0:
            enable_grad(self.unet_trainable_params, False)
            error_neg = self.compute_error(noisy_latent, timesteps, hiddens_neg, masks_neg, target)
            loss_neg = -error_neg
            enable_grad(self.unet_trainable_params, True)
        else:
            error_neg = torch.zeros(bs, device=device)
            loss_neg = torch.zeros(bs, device=device)
        # ----------------------------------------------------------------------------------------

        # ------------------------------ positive loss -------------------------------------------
        error_pos = self.compute_error(noisy_latent, timesteps, hiddens_pos, masks_pos, target)
        loss_pos = error_pos
        # ----------------------------------------------------------------------------------------

        loss = weight_contrastive * loss_neg + loss_pos

        return {
            "loss": loss,
            "error_neg": error_neg,
            "error_pos": error_pos,
            "loss_neg": loss_neg,
            "loss_pos": loss_pos,
        }

    def sample(self, output_slots, resolution: int = 256, seed: int = 1234,
               guidance_scale: int = None, hidden_states=None):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        device = self.device
        batch_size = output_slots["slots"].shape[0]
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale

        unet = self.unet
        vae = self.vae

        scheduler = self.scheduler_test
        scheduler.set_timesteps(self.inference_steps, device=device)

        latents = scheduler.init_noise_sigma * torch.randn(
            (batch_size, unet.config.in_channels, resolution // 8, resolution // 8),
            generator=generator,
        ).to(device)

        if hidden_states is None:
            hidden_states = self.encoder.compute_hidden_states(output_slots)
        attention_mask = None

        if guidance_scale is not None:
            negative_embeds = self.encoder.negative_prompt(batch_size)
            seq_len = max(hidden_states.shape[1], negative_embeds.shape[1])

            negative_embeds, negative_masks = pad_embeddings(negative_embeds, seq_len)
            hidden_states, attention_mask = pad_embeddings(hidden_states, seq_len)

            hidden_states = torch.cat((negative_embeds, hidden_states))
            attention_mask = torch.cat((negative_masks, attention_mask))

        with torch.autocast(device_type="cuda", dtype=self.weight_dtype):

            for t in scheduler.timesteps:
                latent_model_input = torch.cat(
                    [latents] * 2) if guidance_scale is not None else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                noise_pred = unet(
                    latent_model_input, t,
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                ).sample

                # perform guidance
                if guidance_scale is not None:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                if guidance_scale is not None and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            latents = latents / vae.config.scaling_factor
            images = vae.decode(latents, return_dict=False, generator=generator)[0]

        return images

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        if self.training_unet:
            self.unet.train(mode)
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the module."""
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state = {k: v for k, v in state.items() if k not in self.excluded_params}
        return state

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        root = Path(save_directory)
        if self.training_unet_cross_attention:
            unet_dir = root / "unet"
            unet_dir.mkdir(parents=True, exist_ok=True)

            state_dict = {}
            for name, param in self.unet.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param

            save_file(state_dict, unet_dir / "diffusion_pytorch_model.safetensors")
            self.unet.save_config(unet_dir)

        self.encoder.save_pretrained(
            save_directory=root / "encoder",
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, save_directory):
        root = Path(save_directory)
        conf = OmegaConf.load(root / "config.yaml")
        model = instantiate(conf)

        if model.training_unet_cross_attention:
            print("Loading unet weights")
            state_dict = load_file(root / "unet" / "diffusion_pytorch_model.safetensors")
            model.unet.load_state_dict(state_dict, strict=False)
            del state_dict

        print("Loading encoder weights")
        state_dict = load_file(root / "encoder" / "diffusion_pytorch_model.safetensors")
        model.encoder.load_state_dict(state_dict, strict=False)

        del state_dict

        return model
