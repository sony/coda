import json
import os
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from hydra.utils import instantiate
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.metric.segmentation import fARI_metric
from src.metric.segmentation import mbo_metric
from src.metric.segmentation import miou_metric
from src.model.pipeline import DiffusionPipeline
from src.utils import helper

logger = get_logger(__name__)


# https://github.com/JindongJiang/latent-slot-diffusion/blob/master/src/eval/eval_utils.py

def hungarian_algorithm(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.
    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.
    The outputs are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """

    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device), indices.to(device)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Computes the cosine similarity between two tensors.
    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.
    Returns:
        The (batched) cosine similarity between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    dot_products = torch.matmul(a, b.transpose(1, 2))
    norm_a = (a * a).sum(dim=2).sqrt().unsqueeze(2)
    norm_b = (b * b).sum(dim=2).sqrt().unsqueeze(1)
    return dot_products / (torch.matmul(norm_a, norm_b) + eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Computes the cosine distance between two tensors, as 1 - cosine_similarity.
    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.
    Returns:
        The (batched) cosine distance between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    return 1 - cosine_similarity(a, b, eps)


def get_mask_cosine_distance(true_mask: torch.Tensor, pred_mask: torch.Tensor):
    """Computes the cosine distance between the true and predicted masks.
    Args:
        true_mask (Tensor): Tensor of shape (batch size, num objects, 1, H, W).
        pred_mask (Tensor): Tensor of shape (batch size, num slots, 1, H, W).
    Returns:
        The (batched) cosine similarity between the true and predicted masks, with
        shape (batch size, num objects, num slots).
    """
    return cosine_distance(true_mask.flatten(2).detach(), pred_mask.flatten(2).detach())


def measure_representation(
    model: nn.Module,
    dataloader,
    accelerator,
    regresion_label_names,
    classification_label_names,
    max_num_obj: int = 23,
    n_samples: int = 5000,
    sample_size: int = 64,
    train_portion: float = 0.83,
    seed: int = 1234,
):
    """
    Borrowed code from https://github.com/JindongJiang/latent-slot-diffusion/blob/master/src/eval/eval.py
    """
    pbar = tqdm(dataloader, ncols=120, disable=not accelerator.is_main_process)
    encoder = accelerator.unwrap_model(model)

    slots_list = []
    label_continuous_dict = {k: [] for k in regresion_label_names}
    label_discrete_dict = {k: [] for k in classification_label_names}

    with torch.no_grad():

        total_data = 0
        for batch_idx, batch in enumerate(pbar):
            pixel_values, true_masks, labels = batch["image"], batch["mask"].long(), batch["labels"]

            slots_output = encoder.encoder(pixel_values)
            attns, slots = slots_output["attn"], slots_output["slots"]

            attns = rearrange(attns, "b 1 (h w) n -> b n h w", h=sample_size, w=sample_size)
            attns = F.interpolate(
                attns,
                true_masks.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

            attn_argmax = attns.argmax(dim=1, keepdim=False)
            attns_one_hot = F.one_hot(attn_argmax, num_classes=attns.shape[1]).float()
            attns_one_hot = rearrange(attns_one_hot, 'b h w num_slots -> b num_slots h w')

            true_masks_one_hot = F.one_hot(true_masks, num_classes=max_num_obj + 1).float()
            true_masks_one_hot = rearrange(true_masks_one_hot, 'b h w c -> b c h w')

            cost_matrix = get_mask_cosine_distance(
                true_masks_one_hot[..., None, :, :],
                attns_one_hot[..., None, :, :]
            )  # attns or attns_one_hot

            if labels['visibility'].shape[1] >= max_num_obj + 1:
                selected_objects = labels['visibility']
                selected_objects[:, 0] = 0
                if len(selected_objects.shape) == 2:
                    selected_objects = selected_objects[:, :, None]
                obj_idx_adjustment = 0  # multi_object_datasets
            else:
                objects = torch.zeros_like(labels['visibility'][:, 0:1]).to(
                    labels['visibility'].device)
                selected_objects = torch.cat([objects, labels['visibility'] > 0], dim=1)[..., None]
                obj_idx_adjustment = 1  # movi series or clevrtex

            cost_matrix = cost_matrix * selected_objects + 10000 * (1 - selected_objects)
            _, indices = hungarian_algorithm(cost_matrix)

            for idx_in_batch, num_o in enumerate(labels['num_obj']):

                for gt_idx, pred_idx in zip(indices[idx_in_batch][0], indices[idx_in_batch][1]):
                    if selected_objects[idx_in_batch, ..., 0][gt_idx] == 0:
                        # no gt_idx - 1 here because we added the background to the beginning
                        continue

                    # slot_obj = slots[idx_in_batch, 0, pred_idx]
                    slot_obj = slots[idx_in_batch, pred_idx]
                    slots_list.append(slot_obj)
                    for k in label_continuous_dict.keys():
                        l = labels[k][idx_in_batch, gt_idx - obj_idx_adjustment]
                        label_continuous_dict[k].append(l)
                    for k in label_discrete_dict.keys():
                        l = (labels[k][idx_in_batch, gt_idx - obj_idx_adjustment]).long()
                        label_discrete_dict[k].append(l)

            total_data += pixel_values.shape[0]

            if total_data >= n_samples:
                break

        slots_list = torch.stack(slots_list, dim=0)

        # for training use float32 only
        weight_dtype = torch.float32
        slots_list = slots_list.to(device=accelerator.device, dtype=weight_dtype)

        label_continuous_dict = {
            k: torch.stack(v, dim=0).to(device=accelerator.device, dtype=weight_dtype)
            for k, v in label_continuous_dict.items()
        }

        for k, v in label_continuous_dict.items():
            label_continuous_dict[k] = v

        label_continuous_dict = {
            k: v[..., None] if len(v.shape) == 1 else v for k, v in label_continuous_dict.items()
        }

        label_discrete_dict = {
            k: torch.stack(v, dim=0).to(device=accelerator.device, dtype=torch.long)
            for k, v in label_discrete_dict.items()
        }
        label_discrete_dict = {
            k: v[..., 0] if len(v.shape) == 2 else v
            for k, v in label_discrete_dict.items()
        }

    slot_continuous_net_dict = {
        k: nn.Sequential(
            nn.Linear(slots_list.shape[1], slots_list.shape[1]),
            nn.ReLU(),
            nn.Linear(slots_list.shape[1], v.shape[1]),
        ).to(device=accelerator.device, dtype=weight_dtype)
        for k, v in label_continuous_dict.items()
    }

    slot_discrete_net_dict = {
        k: nn.Linear(slots_list.shape[1], int(v.max() + 1)
                     ).to(device=accelerator.device, dtype=weight_dtype)
        for k, v in label_discrete_dict.items()
    }

    # shuffle data and label in the same way
    torch.random.manual_seed(seed)
    shuffle_idx = torch.randperm(slots_list.shape[0])
    for k, v in label_continuous_dict.items():
        label_continuous_dict[k] = label_continuous_dict[k][shuffle_idx]
    for k, v in label_discrete_dict.items():
        label_discrete_dict[k] = label_discrete_dict[k][shuffle_idx]
    slots_list = slots_list[shuffle_idx]
    slots_list_train = slots_list[:int(slots_list.shape[0] * train_portion)]
    slots_list_test = slots_list[int(slots_list.shape[0] * train_portion):]

    label_continuous_dict_train = {
        k: v[:int(v.shape[0] * train_portion)] for k, v in label_continuous_dict.items()}
    label_continuous_dict_test = {
        k: v[int(v.shape[0] * train_portion):] for k, v in label_continuous_dict.items()}
    label_discrete_dict_train = {
        k: v[:int(v.shape[0] * train_portion)] for k, v in label_discrete_dict.items()}
    label_discrete_dict_test = {
        k: v[int(v.shape[0] * train_portion):] for k, v in label_discrete_dict.items()}

    params = []
    for k, v in slot_continuous_net_dict.items():
        params = params + list(v.parameters())
    for k, v in slot_discrete_net_dict.items():
        params = params + list(v.parameters())

    optimizer = torch.optim.AdamW(params)
    slot_continuous_loss = dict()
    slot_discrete_loss = dict()

    print('start property prediction training')

    num_training_steps = 4000

    progress_bar = tqdm(
        range(0, num_training_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        position=0, leave=True
    )

    for _ in range(num_training_steps):
        optimizer.zero_grad()
        for k, v in slot_continuous_net_dict.items():
            slot_continuous_loss[k] = F.mse_loss(
                v(slots_list_train), label_continuous_dict_train[k])
        for k, v in slot_discrete_net_dict.items():
            slot_discrete_loss[k] = F.cross_entropy(
                v(slots_list_train), label_discrete_dict_train[k])
        loss = sum(slot_continuous_loss.values()) + \
            sum(slot_discrete_loss.values())
        loss = loss / slots_list_train.shape[0]
        loss.backward()
        optimizer.step()
        progress_bar.update(1)

    all_loss = dict()
    for k, v in slot_continuous_net_dict.items():
        pred, target = v(slots_list_test), label_continuous_dict_test[k]
        loss = F.mse_loss(pred, target).item()
        all_loss[k] = loss
        print(f'continuous {k}:', all_loss[k])

    for k, v in slot_discrete_net_dict.items():
        pred = torch.argmax(v(slots_list_test), dim=1)
        target = label_discrete_dict_test[k]
        loss = (pred == target).float().mean().item()

        all_loss[k] = loss
        print(f'discrete {k}:', all_loss[k])

    return all_loss


@torch.no_grad()
def measure_segmentation(
    model: nn.Module,
    dataloader,
    accelerator,
    n_samples: int = 5000,
    sample_size: int = 64,
    save_output: str = None,
):
    pbar = tqdm(dataloader, ncols=120, disable=not accelerator.is_main_process)
    encoder = accelerator.unwrap_model(model)
    report_sem_mask = False

    if save_output is not None:
        save_output = Path(save_output)
        save_output.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    fARI, mbo, miou, smbo, smiou = 0, 0, 0, 0, 0

    for i, batch in enumerate(pbar):
        image, mask_true = batch["image"], batch["mask"].long()
        attn = encoder.encoder(image)["attn"]

        inst_overlap_mask = batch.get("inst_overlap_mask", None)
        sem_mask_true = batch.get("sem_mask", None)

        attn = rearrange(attn, "b 1 (h w) n -> b n h w", h=sample_size, w=sample_size)
        mask_pred = F.interpolate(
            attn,
            mask_true.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).argmax(dim=1)

        if i == 0 and accelerator.is_main_process and (save_output is not None):
            rgb_mask_true = helper.draw_rgb_mask(image, mask_true, 0.5)
            rgb_mask_pred = helper.draw_rgb_mask(image, mask_pred, 0.5)
            samples = torch.cat([image, rgb_mask_true, rgb_mask_pred], axis=0)

            img_name = save_output / f"sample.png"
            save_image(samples*0.5 + 0.5, fp=img_name, nrow=image.shape[0])

        mask_pred = accelerator.gather_for_metrics(mask_pred)
        mask_true = accelerator.gather_for_metrics(mask_true)

        if inst_overlap_mask is not None:
            inst_overlap_mask = accelerator.gather_for_metrics(inst_overlap_mask)

        if sem_mask_true is not None:
            sem_mask_true = accelerator.gather_for_metrics(sem_mask_true.long())
            report_sem_mask = True

        missing = n_samples - total_samples
        if missing < mask_pred.shape[0]:
            mask_pred = mask_pred[:missing]
            mask_true = mask_true[:missing]
            if sem_mask_true is not None:
                sem_mask_true = sem_mask_true[:missing]

        batch_size = mask_pred.shape[0]
        total_samples += batch_size

        fARI += float(fARI_metric(mask_true, mask_pred, inst_overlap_mask) * batch_size)
        mbo += float(mbo_metric(mask_true, mask_pred, inst_overlap_mask) * batch_size)
        miou += float(miou_metric(mask_true, mask_pred, inst_overlap_mask) * batch_size)

        if sem_mask_true is not None:
            smbo += float(mbo_metric(sem_mask_true, mask_pred, inst_overlap_mask) * batch_size)
            smiou += float(miou_metric(sem_mask_true, mask_pred, inst_overlap_mask) * batch_size)

        pbar.set_description(f"Samples={total_samples}")

        if total_samples == n_samples:
            break

    result = {"fARI": fARI / total_samples, "MBO": mbo /
              total_samples, "MIOU": miou / total_samples}

    if report_sem_mask:
        result["sMBO"] = smbo / total_samples
        result["sMIOU"] = smiou / total_samples

    logger.info(f"Segmentation: sample={total_samples}", main_process_only=True)
    logger.info(result, main_process_only=True)

    return result


@hydra.main(version_base=None, config_path="./setup", config_name="prob.yaml")
def main(hp: omegaconf.DictConfig):
    """Run the main application.

    Args:
        config (omegaconf.DictConfig): Detailed configurations.
    """
    project_dir = helper.get_run_dir()
    accelerator = Accelerator(
        mixed_precision="no",
        project_dir=project_dir,
    )

    model = DiffusionPipeline.from_pretrained(hp.model_path)

    helper.set_grad(model, False)
    model.to(accelerator.device)
    model.eval()

    # make a different seed for each process
    set_seed(hp.seed + accelerator.process_index)

    dataset = instantiate(hp.dataset.test)
    dataloader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        num_workers=hp.n_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn() if hasattr(dataset, "collate_fn") else None,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    summary = {}
    result = measure_segmentation(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        n_samples=len(dataset) if not hp.get("debug", False) else 2,
        sample_size=hp.sample_size,
    )
    summary = {**result, **summary}

    if hp.dataset.name == "clevrtex":
        classification_label_names = ["material", "shape",]
        regresion_label_names = ["pixel_coords"]
    elif ("movi-e" == hp.dataset.name) or ("movi-c" == hp.dataset.name):
        classification_label_names = ["category"]
        regresion_label_names = ["image_positions", "bboxes_3d"]
    else:
        raise Exception(f"{hp.dataset.name} is not supported")

    result = measure_representation(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        classification_label_names=classification_label_names,
        regresion_label_names=regresion_label_names,
        max_num_obj=hp.dataset.test.max_num_obj,
        n_samples=len(dataset),  # if not hp.get("debug", False) else 1000,
        sample_size=hp.sample_size,
        train_portion=hp.train_portion,
        seed=hp.seed,
    )
    summary = {**result, **summary}

    if accelerator.is_main_process:
        with open(project_dir / "summary.json", "w", encoding='utf-8') as writer:
            json.dump(summary, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
