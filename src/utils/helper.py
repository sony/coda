import os
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from safetensors.torch import _remove_duplicate_names
from scipy.optimize import linear_sum_assignment
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks

COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
    '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080',
    '#C56932', '#b7a58c', '#3a627d', '#9abc15',
    '#54810c', '#a7389c', '#687253', '#61f584',
    '#9a17d4', '#52b0c1', '#21f5b4', '#a2856c',
    '#9b1c34', '#4b1062', '#7cf406', '#0b1f63',
]*5


# https://github.com/Stability-AI/generative-models/issues/406
def load_model_from_network_storage(checkpoint_path, device):
    """
    Load a safetensors model from network storage by first copying to memory

    Args:
        checkpoint_path: Path to the safetensors file
    """
    print(f"Costumized loading from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        file_content = f.read()

    # Load using safetensors.torch.load
    try:
        tensors = safetensors.torch.load(file_content)
        tensors = {k: v.to(device) for k, v in tensors.items()}
        return tensors
    except Exception as e:
        print(f"Error loading tensors: {str(e)}")
        raise


def accelerate_load_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    strict: bool = True,
    device: Union[str, int] = "cpu",
):
    """
    Loads a given filename onto a torch model.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to load onto.
        filename (`str`, or `os.PathLike`):
            The filename location to load the file from.
        strict (`bool`, *optional*, defaults to True):
            Whether to fail if you're missing keys or having unexpected ones.
            When false, the function simply returns missing and unexpected names.
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `(missing, unexpected): (List[str], List[str])`
            `missing` are names in the model which were not modified during loading
            `unexpected` are names that are on the file, but weren't used during
            the load.
    """
    state_dict = load_model_from_network_storage(filename, device=device)
    # state_dict = load_file(filename, device=device)

    model_state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)

    if strict and (missing or unexpected):
        missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
        unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
        error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
        if missing:
            error += f"\n    Missing key(s) in state_dict: {missing_keys}"
        if unexpected:
            error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
        raise RuntimeError(error)

    return missing, unexpected


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_run_dir():
    """Return the working directory of hydra."""
    return Path(to_absolute_path(HydraConfig.get().run.dir))


def numel(model: nn.Module):
    """
    Return number of parameters.

    Args:
        model (nn.Module): Input model.

    Returns:
        int: Total number of parameters.
        int: Trainable parameters.
        int: Non-trainable parameters.
    """
    total, trainable, non_trainable = 0, 0, 0
    for p in model.parameters():
        total += np.prod(p.size())
        if p.requires_grad:
            trainable += np.prod(p.size())
        else:
            non_trainable += np.prod(p.size())

    return total, trainable, non_trainable


def broadcast_shapes(src: torch.Tensor, target_shape: torch.Size):
    """Make sure the dimensions of `src` matches `target_shape` by adding or removing
    dimension from the right.

    Args:
        source (torch.Tensor): Source Tensor.
        target (torch.Size): Destination shape.

    Returns:
        torch.Tensor: Returned tensor.
    """
    src_shape = src.shape[:len(target_shape)]
    while len(src_shape) < len(target_shape):
        src_shape += (1,)

    return src.reshape(src_shape)


def mean_flat(x: torch.Tensor):
    """Take the mean over all non-batch dimensions."""
    return x.mean(dim=list(range(1, len(x.shape))))


def sum_flat(x: torch.Tensor):
    """Take the sum over all non-batch dimensions."""
    return x.sum(dim=list(range(1, len(x.shape))))


def collate_fn(hp):

    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

    if "train" in hp.dataset:
        train_ds = instantiate(hp.dataset.train)
        train_dataloader = DataLoader(
            train_ds,
            batch_size=hp.trainer.batch_size,
            num_workers=hp.trainer.n_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
        )

    if "valid" in hp.dataset:
        eval_ds = instantiate(hp.dataset.valid)
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=hp.trainer.batch_size,
            num_workers=hp.trainer.n_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=eval_ds.collate_fn() if hasattr(eval_ds, "collate_fn") else None,
        )

    if "test" in hp.dataset:
        test_ds = instantiate(hp.dataset.test)
        test_dataloader = DataLoader(
            test_ds,
            batch_size=hp.trainer.batch_size,
            num_workers=hp.trainer.n_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=test_ds.collate_fn() if hasattr(test_ds, "collate_fn") else None,
        )

    return train_dataloader, eval_dataloader, test_dataloader


def cycle_dataloader(dataloader):
    """Make an infinite itreation of dataloader.

    Args:
        dataloader (DataLoaderShard): Dataloader from accelerator.

    NOTE: set_epoch will be called automatically after iterating over the dataloader.
    """
    while True:
        for data in dataloader:
            yield data


def set_grad(module, enable_grad=True):
    for p in module.parameters():
        p.requires_grad_(enable_grad)


def load_model(pretrained_weight_path: str, config_path: str):
    """Load model from a checkpoint.

    Args:
        pretrained_weight_path (str): Path to pretrained weights.
        config_path (str): Path to configuration.

    Returns:
        nn.Module: Loaded model.
        DataLoader: Training DataLoader.
        DataLoader: Test DataLoader.
    """
    checkpoint = torch.load(pretrained_weight_path, map_location="cpu", weights_only=False)
    conf = OmegaConf.load(config_path)

    print(f"Loading model from {pretrained_weight_path}")

    model = instantiate(conf.pipeline)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    del checkpoint

    return model, conf


def draw_rgb_mask(rgb, mask, alpha=0.7):
    """Map mask to rgb image."""
    assert 0. < alpha < 1.

    masks = F.one_hot(mask) > 0
    masks = rearrange(masks, "b h w c -> b c h w")

    output = []
    for image, mask in zip(rgb * 0.5 + 0.5, masks):
        image = draw_segmentation_masks(image, mask, alpha=alpha, colors=COLORS)
        output.append(image)

    return torch.stack(output) * 2.0 - 1.0


def triu_flatten(tril):
    N = tril.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indicies = N * indicies[0] + indicies[1]
    return tril.flatten(-2)[..., indicies]


def pairwise_cosine_distance(X: torch.Tensor, Y: torch.Tensor):
    """Cosine distance

    Args:
        X (torch.Tensor): Tensor of shape (B, N, D).
        Y (torch.Tensor): Tensor of shape (B, N, D).

    Returns:
        torch.Tensor: Output of shape (B, N, N).
    """

    X = F.normalize(X, p=2, dim=-1)
    Y = F.normalize(Y, p=2, dim=-1)

    return 2 - 2*torch.bmm(X, Y.transpose(1, 2))


def assignment(source: torch.Tensor, target: torch.Tensor):
    """Run linear asigmment for batch of tensors.

    Args:
        source (torch.Tensor): Tensor of shape (B, N, D).
        target (torch.Tensor): Tensor of shape (B, N, D).

    Returns:
        torch.Tensor: The matched tensor for source of shape (B, N, D).
    """
    output = []
    # compute cosine distance
    with torch.no_grad():
        cosine_dist = pairwise_cosine_distance(source, target).detach().cpu().numpy()

    for cost, col in zip(cosine_dist, target):
        index = linear_sum_assignment(cost)[1]
        output.append(col[index])

    return torch.stack(output)
