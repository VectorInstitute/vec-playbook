"""Fine-tune a HF model for text classification with a basic loop."""

import os

import submitit
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class TextClassificationTrainer(submitit.helpers.Checkpointable):
    """Trainer for text classification."""

    def __init__(self):
        """Initialize the trainer."""
        self.ckpt_dir = None

    def _latest_checkpoint(self, out_dir):
        names = [n for n in os.listdir(out_dir) if n.startswith("checkpoint-")]
        if not names:
            return None
        steps = sorted(int(n.split("-")[1]) for n in names if "-" in n)
        return os.path.join(out_dir, f"checkpoint-{steps[-1]}") if steps else None

    def __call__(self, cfg):
        """Train the model."""
        out_dir = os.path.join(cfg.work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        self.ckpt_dir = self._latest_checkpoint(out_dir)

        model_name = getattr(cfg, "model_name", "distilbert-base-uncased")
        ds = load_dataset("ag_news")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        def tok_fn(ex):
            return tok(
                ex["text"], truncation=True, max_length=getattr(cfg, "max_length", 256)
            )

        ds = ds.map(tok_fn, batched=True)
        collator = DataCollatorWithPadding(tokenizer=tok)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=getattr(cfg, "num_labels", 4)
        )

        args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=getattr(cfg, "num_train_epochs", 2),
            per_device_train_batch_size=getattr(cfg, "per_device_train_batch_size", 16),
            per_device_eval_batch_size=getattr(cfg, "per_device_eval_batch_size", 32),
            evaluation_strategy="steps",
            eval_steps=getattr(cfg, "eval_steps", 200),
            logging_steps=getattr(cfg, "logging_steps", 50),
            learning_rate=getattr(cfg, "learning_rate", 5e-5),
            weight_decay=getattr(cfg, "weight_decay", 0.01),
            save_strategy="steps",
            save_steps=getattr(cfg, "save_steps", 100),  # checkpoints every N steps
            save_total_limit=getattr(cfg, "save_total_limit", 2),
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            tokenizer=tok,
            data_collator=collator,
        )

        trainer.train(resume_from_checkpoint=self.ckpt_dir)
        metrics = trainer.evaluate()
        print(metrics)
        return 0

    def checkpoint(self, *args, **kwargs):
        """Checkpoint the trainer."""
        # Trainer already wrote step checkpoint(s) to self.ckpt_dir/output_dir.
        # Returning a DelayedSubmission will requeue the same callable.
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
