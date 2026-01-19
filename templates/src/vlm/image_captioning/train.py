"""Fine-tune BLIP VLM model for image captioning."""

import logging
import os
import random

import submitit
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor


logger = logging.getLogger(__name__)


class ImageCaptioningTrainer(submitit.helpers.Checkpointable):
    """Trainer for VLM image captioning."""

    def __init__(self):
        """Initialize the trainer."""
        self.ckpt_dir = None

    def _latest_checkpoint(self, out_dir):
        """Find the latest checkpoint directory."""
        if not os.path.exists(out_dir):
            return None
        names = [n for n in os.listdir(out_dir) if n.startswith("checkpoint-epoch-")]
        if not names:
            return None
        epochs = sorted(
            int(n.split("-")[-1]) for n in names if n.split("-")[-1].isdigit()
        )
        return (
            os.path.join(out_dir, f"checkpoint-epoch-{epochs[-1]}") if epochs else None
        )

    def _setup_data(self, processor, cfg):
        """Set up dataset and dataloaders."""
        dataset_name = OmegaConf.select(cfg, "trainer.dataset_name", default="cifar10")
        batch_size = OmegaConf.select(cfg, "trainer.batch_size", default=16)

        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name == "cifar10":
            ds = load_dataset("cifar10")
            logger.info(f"Dataset splits: {list(ds.keys())}")

            cifar_classes = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]

            def add_captions(example):
                label = example["label"]
                caption = f"a photo of a {cifar_classes[label]}"
                example["caption"] = caption
                return example

            train_split = ds["train"].map(add_captions)
            eval_split = ds["test"].map(add_captions)

            image_col = "img"
            text_col = "caption"

        else:
            ds = load_dataset(dataset_name)
            logger.info(f"Dataset splits: {list(ds.keys())}")

            train_split = ds.get("train") or ds[list(ds.keys())[0]]
            eval_split = ds.get("validation") or ds.get("test")

            logger.info(f"Train split columns: {train_split.column_names}")
            logger.info(f"Sample train item: {train_split[0]}")

            image_col = None
            text_col = None

            for col in train_split.column_names:
                if any(keyword in col.lower() for keyword in ["image", "img", "photo"]):
                    image_col = col
                elif any(
                    keyword in col.lower()
                    for keyword in ["text", "caption", "label", "title"]
                ):
                    text_col = col

            if image_col is None or text_col is None:
                raise ValueError(
                    f"Could not find image and text columns. Available columns: {train_split.column_names}"
                )

        logger.info(f"Using columns: image='{image_col}', text='{text_col}'")

        def collate(batch):
            images = [b[image_col] for b in batch]
            texts = [b[text_col] for b in batch]
            enc = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            enc["labels"] = enc["input_ids"].clone()
            return enc

        train_loader = DataLoader(
            train_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate,
        )
        eval_loader = (
            DataLoader(
                eval_split,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate,
            )
            if eval_split is not None
            else None
        )
        return train_loader, eval_loader

    def _train_epoch(
        self, model, optimizer, train_loader, epoch, device, cfg, global_step
    ):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        seen = 0
        print_every = OmegaConf.select(cfg, "trainer.print_every", default=100)

        for batch in train_loader:
            batch_device = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            out = model(
                pixel_values=batch_device["pixel_values"],
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            loss = out.loss
            loss.backward()
            optimizer.step()

            bs = batch["pixel_values"].size(0)
            running_loss += loss.item() * bs
            seen += bs
            global_step += 1

            if print_every > 0 and (global_step % print_every == 0):
                avg_loss = running_loss / max(seen, 1)
                logger.info(f"Epoch {epoch} Step {global_step}: loss={avg_loss:.4f}")
                running_loss = 0.0
                seen = 0

        return global_step

    @torch.no_grad()
    def _evaluate_epoch(self, model, eval_loader, epoch, device):
        """Evaluate for one epoch."""
        if eval_loader is None:
            return

        model.eval()
        total = 0
        loss_sum = 0.0

        for batch in eval_loader:
            batch_device = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            out = model(
                pixel_values=batch_device["pixel_values"],
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                labels=batch_device["labels"],
            )
            bs = batch["pixel_values"].size(0)
            loss_sum += out.loss.item() * bs
            total += bs

        if total > 0:
            avg_loss = loss_sum / total
            logger.info(f"Epoch {epoch} eval: loss={avg_loss:.4f}")

    def _save_checkpoint(self, model, processor, epoch, out_dir):
        """Save model and processor checkpoint."""
        save_dir = os.path.join(out_dir, f"checkpoint-epoch-{epoch}")
        logger.info(f"Saving checkpoint-epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

    def __call__(self, cfg):
        """Train the VLM model."""
        cfg: DictConfig = OmegaConf.create(cfg)  # Ensure cfg is a DictConfig

        # Create output directory
        out_dir = cfg.paths.out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Get ckpt dir
        self.ckpt_dir = self._latest_checkpoint(out_dir)

        # Configuration
        model_name = OmegaConf.select(
            cfg, "trainer.model_name", default="Salesforce/blip-image-captioning-base"
        )
        lr = OmegaConf.select(cfg, "trainer.learning_rate", default=1e-5)
        num_epochs = OmegaConf.select(cfg, "trainer.num_epochs", default=2)
        seed = OmegaConf.select(cfg, "trainer.seed", default=42)

        # Set seed
        logger.info(f"Starting VLM captioning training with seed {seed}")
        random.seed(seed)
        torch.manual_seed(seed)

        # Setup device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = BlipProcessor.from_pretrained(model_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Setup data
        train_loader, eval_loader = self._setup_data(processor, cfg)

        # Resume from checkpoint if available
        start_epoch = 0
        if self.ckpt_dir and os.path.exists(self.ckpt_dir):
            logger.info(f"Resuming from checkpoint: {self.ckpt_dir}")
            model = BlipForConditionalGeneration.from_pretrained(self.ckpt_dir).to(
                device
            )
            processor = BlipProcessor.from_pretrained(self.ckpt_dir)
            start_epoch = int(os.path.basename(self.ckpt_dir).split("-")[-1]) + 1

        # Training loop
        global_step = 0
        for epoch in range(start_epoch, num_epochs):
            global_step = self._train_epoch(
                model, optimizer, train_loader, epoch, device, cfg, global_step
            )
            self._evaluate_epoch(model, eval_loader, epoch, device)
            self._save_checkpoint(model, processor, epoch, out_dir)

        logger.info("Training completed!")
        return 0

    def checkpoint(self, *args, **kwargs):
        """Checkpoint the trainer."""
        # Model checkpoints are already saved in _save_checkpoint.
        # Returning a DelayedSubmission will requeue the same callable.
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
