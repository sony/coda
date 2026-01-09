import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models import ModelMixin
from einops import rearrange
from torch import nn
from transformers import CLIPTextModel
from transformers import CLIPTokenizer

from ..layers.backbone import DINOEncoder
from ..layers.slot_attn import SlotAttn


def get_negative_prompt_embeds(model_name: str):
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
    )
    uncond_input = tokenizer(
        [""], padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    return text_encoder(uncond_input.input_ids)[0].detach()


@torch.no_grad()
def get_negative_prompt_embeds_sd3(model_name: str):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_name)
    _, negative_prompt_embeds, _, negative_pooled_prompt_embeds = pipe.encode_prompt("", None, None)
    return negative_prompt_embeds.detach(), negative_pooled_prompt_embeds.detach()


class LatentSlotDiffusion(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        model_id: str,
        dino_id: str,
        dino_requires_grad: bool,
        dino_sample_size: int,
        dino_out_channels: int,
        slot_n_iterations: int,
        slot_n_slots: int,
        slot_n_heads: int,
        slot_out_channels: int,
        slot_size: int,
        slot_mlp_hidden_size: int,
        slot_epsilon: float = 1e-8,
        slot_learnable_slot_init: bool = False,
        slot_bi_level: bool = False,
        dino_enable_register: bool = False,
    ):
        super().__init__()

        self.backbone = DINOEncoder(
            model_name=dino_id,
            requires_grad=dino_requires_grad,
            sample_size=dino_sample_size,
            out_channels=dino_out_channels,
            dino_enable_register=dino_enable_register,
        )

        self.slot_attn = SlotAttn(
            num_iterations=slot_n_iterations,
            num_slots=slot_n_slots,
            num_heads=slot_n_heads,
            input_size=dino_out_channels,
            out_size=slot_out_channels,
            slot_size=slot_size,
            mlp_hidden_size=slot_mlp_hidden_size,
            input_resolution=dino_sample_size,
            epsilon=slot_epsilon,
            learnable_slot_init=slot_learnable_slot_init,
            bi_level=slot_bi_level,
        )

        null_embedding = get_negative_prompt_embeds(model_id).detach()
        self.register_buffer("null_embedding", null_embedding)

    def compute_hidden_states(self, output_slots):
        return output_slots["slots"]

    def negative_prompt(self, batch_size: int):
        """Returns negative prompt embeddings."""
        embeds = self.null_embedding
        return embeds.repeat((batch_size, 1, 1))

    def forward(self, x: torch.Tensor, eps: torch.Tensor = None):
        """Given an image x, return slots and attention masks.

        Args:
            x (torch.Tensor): Input image of shape (B, 3, H, W).

        Returns:
            Dict: 
                `slots`: A tensor of shape (B, N, D).
                `attn`: An attention mask of shape (B, n_heads, M, N).
                    where M is the number of input tokens.
        """

        latent_code = self.backbone(x)
        latent_code = rearrange(latent_code, "b d h w -> b 1 d h w")

        slots, attn = self.slot_attn(latent_code, eps)

        slots = rearrange(slots, "b 1 n d -> b n d")
        attn = rearrange(attn, "b 1 h m n -> b h m n")

        return {"slots": slots, "attn": attn}


class RegisterSlotDiffusion(LatentSlotDiffusion):

    @register_to_config
    def __init__(
        self,
        model_id: str,
        dino_id: str,
        dino_requires_grad: bool,
        dino_sample_size: int,
        dino_out_channels: int,
        slot_n_iterations: int,
        slot_n_slots: int,
        slot_n_heads: int,
        slot_out_channels: int,
        slot_size: int,
        slot_mlp_hidden_size: int,
        slot_epsilon: float = 1e-8,
        slot_learnable_slot_init: bool = False,
        slot_bi_level: bool = False,
        num_registers: int = 0,
        dino_enable_register: bool = False,
    ):

        super().__init__(
            model_id=model_id,
            dino_id=dino_id,
            dino_requires_grad=dino_requires_grad,
            dino_sample_size=dino_sample_size,
            dino_out_channels=dino_out_channels,
            slot_n_iterations=slot_n_iterations,
            slot_n_slots=slot_n_slots,
            slot_n_heads=slot_n_heads,
            slot_out_channels=slot_out_channels,
            slot_size=slot_size,
            slot_mlp_hidden_size=slot_mlp_hidden_size,
            slot_epsilon=slot_epsilon,
            slot_learnable_slot_init=slot_learnable_slot_init,
            slot_bi_level=slot_bi_level,
            dino_enable_register=dino_enable_register,
        )

        if num_registers > 0:
            # register tokens are padded to slots
            self.register_tokens = nn.Parameter(torch.rand((1, num_registers, slot_out_channels)))
            nn.init.xavier_uniform_(self.register_tokens)
        elif num_registers == 0:
            # using learnable token for negative prompt
            self.negative_tokens = nn.Parameter(torch.rand((1, slot_n_slots, slot_out_channels)))
            nn.init.xavier_uniform_(self.negative_tokens)

        self.num_registers = num_registers

    def compute_hidden_states(self, output_slots):
        hidden_states = output_slots["slots"]
        if self.num_registers > 0:
            register_tokens = self.register_tokens.repeat((hidden_states.shape[0], 1, 1))
            hidden_states = torch.cat([hidden_states, register_tokens], dim=1)
        elif self.num_registers < 0:
            null_embedding = self.null_embedding

            if self.num_registers == -2:
                null_embedding = null_embedding[:, hidden_states.shape[1]:, :]

            null_embedding = null_embedding.repeat((hidden_states.shape[0], 1, 1))

            hidden_states = torch.cat([hidden_states, null_embedding], dim=1)

        return hidden_states

    def negative_prompt(self, batch_size: int):
        """Returns negative prompt embeddings."""
        if self.num_registers > 0:
            embeds = self.register_tokens
        elif self.num_registers == 0:
            embeds = self.negative_tokens
        else:
            embeds = self.null_embedding
        return embeds.repeat((batch_size, 1, 1))
