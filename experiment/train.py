
import hydra
import numpy as np
import omegaconf
import torch
import wandb
from omegaconf import OmegaConf
from torchvision.transforms import Resize
from torchvision.utils import make_grid

from experiment.eval import measure_reconstruction
from experiment.eval import measure_segmentation
from src.utils.trainer import Trainer


class VisionTrainer(Trainer):

    @torch.no_grad()
    def valid_one_epoch_hook(self, model, **kargs):
        """Write some impages to the output."""
        hp = self.hp
        accelerator = self.accelerator
        seed = hp.seed
        resolution = hp.dataset.train.img_size

        model = accelerator.unwrap_model(self.model)

        if accelerator.is_main_process:
            random_batch_idx = np.random.randint(0, len(self.eval_dl) - 1)
            for i, batch in enumerate(self.eval_dl):
                if i == random_batch_idx:
                    sample = batch["image"][:8]
                    output_slots = model.encoder(sample)
                    break

            sample_rec = model.sample(output_slots, resolution=resolution, seed=seed)
            images = torch.cat([sample.detach().cpu(), sample_rec.detach().cpu()], axis=0)
            images = ((images + 1.0) / 2.0).clamp(0, 1)

            for tracker in self.accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(kargs["name"], np_images, self.step)
                elif tracker.name == "wandb":
                    images = Resize(128)(images)  # reduce image size to avoid memory
                    wb_images = wandb.Image(make_grid(images, nrow=sample.shape[0]))
                    tracker.log({kargs["name"]: wb_images}, step=self.step,)

        # computing rFID and other reconstruction metrics
        accelerator.wait_for_everyone()
        score_rec = measure_reconstruction(
            model,
            dataloader=self.eval_dl,
            accelerator=self.accelerator,
            n_samples=float("inf") if not hp.get("debug", False) else 2,
            resolution=resolution,
            seed=seed,
        )

        self.write_log(score_rec, f"valid_{kargs['name']}")

        score_seg = {}
        if self.test_dl is not None:
            accelerator.wait_for_everyone()
            # compute segmentation metrics
            score_seg = measure_segmentation(
                model,
                dataloader=self.test_dl,
                accelerator=self.accelerator,
                n_samples=float("inf") if not hp.get("debug", False) else 1,
                sample_size=hp.pipeline.encoder.dino_sample_size,
                save_output=self.project_dir / "samples" / kargs['name'] / f"{self.step:06d}"
            )

            self.write_log(score_seg, f"test_{kargs['name']}")

        # saving the best model
        save_best = False
        inverse_score = -1 if len(score_seg) > 0 else 1
        optimal_score = score_seg if len(score_seg) else {"rFID": score_rec["rFID"]}
        for k, v in optimal_score.items():
            if (k not in self.best_score) or (inverse_score * v < inverse_score * self.best_score[k]):
                self.best_score[k] = v
                save_best = True
                best_metric = k
                best_value = v

        if save_best and accelerator.is_main_process:
            self.accelerator.print(f"Saving `{kargs['name']}` with {best_metric}={best_value:.4f}")
            path = self.project_dir / kargs['name']

            model.save_pretrained(path)
            torch.save({**score_rec, **score_seg, "step": self.step}, path / "summary.pth")
            with open(path / "config.yaml", "w") as writer:
                resolved_cfg = OmegaConf.to_container(hp.pipeline, resolve=True)
                writer.write(OmegaConf.to_yaml(resolved_cfg))


@hydra.main(version_base=None, config_path="./setup", config_name="train.yaml")
def main(hp: omegaconf.DictConfig):
    """Run the main application.

    Args:
        config (omegaconf.DictConfig): Detailed configurations.
    """
    trainer = VisionTrainer(hp)
    trainer.train()


if __name__ == "__main__":
    main()
