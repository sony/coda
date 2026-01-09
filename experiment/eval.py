import json
from pathlib import Path

import hydra
import omegaconf
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from tqdm import tqdm

from src.metric.segmentation import fARI_metric
from src.metric.segmentation import mbo_metric
from src.metric.segmentation import miou_metric
from src.model.pipeline import DiffusionPipeline
from src.utils import helper

logger = get_logger(__name__)


@torch.no_grad()
def measure_reconstruction(
    model: nn.Module,
    dataloader,
    accelerator,
    n_samples: int = 5000,
    resolution: int = 256,
    seed: int = 1234,
):

    fid_evaluator = FrechetInceptionDistance(normalize=True).to(accelerator.device)
    psnr_evaluator = PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
    ssim_evaluator = StructuralSimilarityIndexMeasure().to(accelerator.device)
    lpips_evaluator = LearnedPerceptualImagePatchSimilarity(normalize=True).to(accelerator.device)
    mse = 0
    pbar = tqdm(dataloader, ncols=120, disable=not accelerator.is_main_process)

    unwrap_model = accelerator.unwrap_model(model)
    for batch in pbar:
        real = batch["image"]
        output_slots = unwrap_model.encoder(real)
        fake = unwrap_model.sample(output_slots, resolution=resolution, seed=seed)

        gather_real = accelerator.gather_for_metrics(real)
        gather_fake = accelerator.gather_for_metrics(fake)

        missing = n_samples - fid_evaluator.real_features_num_samples.item()
        if missing < gather_real.shape[0]:
            gather_real = gather_real[:missing]
            gather_fake = gather_fake[:missing]

        gather_real = ((gather_real + 1.) / 2.).clamp(0, 1)
        gather_fake = ((gather_fake + 1.) / 2.).clamp(0, 1)

        fid_evaluator.update(gather_real, real=True)
        fid_evaluator.update(gather_fake, real=False)

        psnr_evaluator.update(gather_fake, gather_real)
        ssim_evaluator.update(gather_fake, gather_real)
        lpips_evaluator.update(gather_fake, gather_real)

        mse += ((gather_real - gather_fake)**2).sum().item()

        pbar.set_description(f"# samples={fid_evaluator.real_features_num_samples.item()}")

        if fid_evaluator.real_features_num_samples.item() == n_samples:
            break

    n_reals = fid_evaluator.real_features_num_samples.item()
    n_fakes = fid_evaluator.fake_features_num_samples.item()

    result = {
        "rFID": fid_evaluator.compute().item(),
        "PSNR": psnr_evaluator.compute().item(),
        "SSIM": ssim_evaluator.compute().item(),
        "LPIPS": lpips_evaluator.compute().item(),
        "MSE": mse / fid_evaluator.fake_features_num_samples.item(),
    }

    logger.info(f"Reconstruction: real={n_reals} fake={n_fakes}", main_process_only=True)
    logger.info(result, main_process_only=True)

    return result


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


@hydra.main(version_base=None, config_path="./setup", config_name="eval.yaml")
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

    model.guidance_scale = hp.guidance_scale
    model.inference_steps = hp.inference_steps

    helper.set_grad(model, False)
    model.to(accelerator.device)
    model.eval()

    # make a different seed for each process
    set_seed(hp.seed + accelerator.process_index)

    dataset = instantiate(hp.dataset.test)
    n_samples = len(dataset)

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
    # ------------------------------------------------------------------------#
    summary = {}
    if hp.report_metrics.reconstruction:
        try:
            result = measure_reconstruction(
                model=model,
                dataloader=dataloader,
                accelerator=accelerator,
                n_samples=n_samples if not hp.get("debug", False) else 2000,
                resolution=hp.dataset.test.img_size,
                seed=hp.seed,
            )
            summary = {**result, **summary}
        except Exception as error:
            logger.info(f"Failed to run reconstruction metrics", main_process_only=True)
            logger.info(f"{error}\n")
            pass
    # ------------------------------------------------------------------------#

    # ------------------------------------------------------------------------#
    if hp.report_metrics.segmentation:
        try:
            result = measure_segmentation(
                model=model,
                dataloader=dataloader,
                accelerator=accelerator,
                n_samples=len(dataset) if not hp.get("debug", False) else 2000,
                sample_size=hp.sample_size,
            )
            summary = {**result, **summary}
        except Exception as error:
            logger.info(f"Failed to run segmentation metrics", main_process_only=True)
            logger.info(f"{error}\n")
            pass

    if accelerator.is_main_process:
        with open(project_dir / "summary.json", "w", encoding='utf-8') as writer:
            json.dump(summary, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
