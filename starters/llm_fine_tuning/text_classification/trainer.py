"""Fine-tune a HF model for text classification with a basic loop."""

from __future__ import annotations

from typing import Any, Optional

import torch
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from vec_tool.core.checkpointable import Checkpointable


class HFTextClassificationTrainer(Checkpointable):
    """Fine-tune a HF model for text classification with a basic loop."""

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        num_epochs: int = 2,
        batch_size: int = 16,
        lr: float = 5e-5,
        print_every: int = 100,
        seed: int = 42,
        **cfg: Any,
    ) -> None:
        super().__init__(**cfg)

        self.ckpt_dir = self.run_dir / "hf_ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = "distilbert-base-uncased"
        self.dataset_name = "ag_news"

        # Hyperparameters
        self.text_column = text_column
        self.label_column = label_column
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.print_every = int(print_every)
        self.seed = int(seed)

    def _setup_data(self, tokenizer):
        """Set up dataset and dataloaders."""
        ds = load_dataset(self.dataset_name)

        def tok(batch):
            return tokenizer(batch[self.text_column], truncation=True)

        ds = ds.map(tok, batched=True, remove_columns=[self.text_column])
        if self.label_column != "labels":
            ds = ds.rename_column(self.label_column, "labels")
        ds = ds.remove_columns(
            [
                c
                for c in ds["train"].column_names
                if c not in {"input_ids", "attention_mask", "labels"}
            ]
        )
        ds.set_format(type="torch")

        train_ds = ds["train"]
        eval_ds = ds.get("validation") or ds.get("test")
        collate = DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate
        )
        eval_loader = (
            DataLoader(
                eval_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate
            )
            if eval_ds is not None
            else None
        )

        return train_ds, train_loader, eval_loader

    def _setup_model(self, train_ds, device):
        """Set up model, optimizer, and loss function."""
        num_labels = int(train_ds["labels"].max().item() + 1)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, criterion

    def _train_epoch(
        self, model, optimizer, criterion, train_loader, device, epoch, global_step
    ):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        for _, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)
            global_step += 1

            if self.print_every > 0 and (global_step % self.print_every == 0):
                avg_loss = running_loss / self.print_every
                acc = 100.0 * correct / max(seen, 1)
                print(
                    f"Epoch {epoch} Step {global_step}: loss={avg_loss:.4f} acc={acc:.2f}%"
                )
                running_loss = 0.0

        return global_step

    def _evaluate_epoch(self, model, criterion, eval_loader, device, epoch):
        """Evaluate for one epoch."""
        if eval_loader is None:
            return

        model.eval()
        eval_correct, eval_seen, eval_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
                eval_loss_sum += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_seen += labels.size(0)
        eval_acc = 100.0 * eval_correct / max(eval_seen, 1)
        print(
            f"Epoch {epoch} eval: loss={eval_loss_sum / max(len(eval_loader), 1):.4f} acc={eval_acc:.2f}%"
        )

    def _save_checkpoint(self, model, tokenizer, epoch):
        """Save model and tokenizer checkpoint."""
        save_dir = self.ckpt_dir / f"checkpoint-epoch-{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    def __call__(self) -> None:
        """Run the main training loop."""
        print(f"Starting training with seed {self.seed}")
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup tokenizer and data
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        train_ds, train_loader, eval_loader = self._setup_data(tokenizer)

        # Setup model
        model, optimizer, criterion = self._setup_model(train_ds, device)

        # Training loop
        start_epoch = int(self._load_step())
        global_step = 0
        for epoch in range(start_epoch, self.num_epochs):
            global_step = self._train_epoch(
                model, optimizer, criterion, train_loader, device, epoch, global_step
            )
            self._evaluate_epoch(model, criterion, eval_loader, device, epoch)
            self._save_checkpoint(model, tokenizer, epoch)
            self._save_step(epoch + 1)

    def _find_last_checkpoint(self) -> Optional[str]:
        if not self.ckpt_dir.exists():
            return None
        ckpts = sorted(
            self.ckpt_dir.glob("checkpoint-epoch-*"), key=lambda p: p.stat().st_mtime
        )
        return str(ckpts[-1]) if ckpts else None
