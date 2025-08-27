"""Fine-tune BLIP VLM model for image captioning."""

from __future__ import annotations, barry_as_FLUFL

import random
from typing import Any, Optional

import torch
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor
from vec_tool.core.checkpointable import Checkpointable


class ImageCaptioningTrainer(Checkpointable):
    """Fine-tune BLIP VLM model for image captioning."""

    def __init__(
        self,
        num_epochs: int = 2,
        batch_size: int = 16,
        lr: float = 1e-5,
        print_every: int = 100,
        seed: int = 42,
        **cfg: Any,
    ) -> None:
        super().__init__(**cfg)
        self.ckpt_dir = self.run_dir / "vlm_caption_ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.dataset_name = "lambdalabs/pokemon-blip-captions"
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.print_every = int(print_every)
        self.seed = int(seed)

    def _setup_data(self, processor):
        ds = load_dataset(self.dataset_name)
        train_split = ds.get("train") or ds[list(ds.keys())[0]]
        eval_split = ds.get("validation") or ds.get("test")

        def collate(batch):
            images = [b["image"] for b in batch]
            texts = [b["text"] for b in batch]
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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate,
        )
        eval_loader = (
            DataLoader(
                eval_split,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate,
            )
            if eval_split is not None
            else None
        )
        return train_loader, eval_loader

    def _setup_model(self, device):
        model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(device)
        processor = BlipProcessor.from_pretrained(self.model_name)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        return model, processor, optimizer

    def _train_epoch(self, model, optimizer, train_loader, epoch, device, global_step):
        model.train()
        running = 0.0
        seen = 0
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
            running += loss.item() * bs
            seen += bs
            global_step += 1
            if self.print_every > 0 and (global_step % self.print_every == 0):
                print(
                    f"Epoch {epoch} Step {global_step}: loss={running / max(seen, 1):.4f}"
                )
                running = 0.0
                seen = 0
        return global_step

    @torch.no_grad()
    def _evaluate_epoch(self, model, eval_loader, epoch, device):
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
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            bs = batch["pixel_values"].size(0)
            loss_sum += out.loss.item() * bs
            total += bs
        if total > 0:
            print(f"Epoch {epoch} eval: loss={loss_sum / total:.4f}")

    def _save_checkpoint(self, model, processor, epoch):
        save_dir = self.ckpt_dir / f"checkpoint-epoch-{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

    def __call__(self) -> None:
        """Train the VLM model for image captioning."""
        print(f"Starting VLM captioning training with seed {self.seed}")
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor, optimizer = self._setup_model(device)
        train_loader, eval_loader = self._setup_data(processor)
        start_epoch = int(self._load_step())
        global_step = 0
        for epoch in range(start_epoch, self.num_epochs):
            global_step = self._train_epoch(
                model, optimizer, train_loader, epoch, device, global_step
            )
            self._evaluate_epoch(model, eval_loader, epoch, device)
            self._save_checkpoint(model, processor, epoch)
            self._save_step(epoch + 1)

    def _find_last_checkpoint(self) -> Optional[str]:
        if not self.ckpt_dir.exists():
            return None
        ckpts = sorted(
            self.ckpt_dir.glob("checkpoint-epoch-*"), key=lambda p: p.stat().st_mtime
        )
        return str(ckpts[-1]) if ckpts else None
