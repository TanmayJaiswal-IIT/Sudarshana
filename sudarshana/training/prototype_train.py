"""Prototype training harness for Sudarshana Token Head.

Note: This is a prototype harness used to validate labels, the token head, and redaction.
P3 owns the final joint training run with pair inputs and normalizer.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..data.loader import load_dataset_split, PromptInjectionDataset
from ..models.token_head import SudarshanaTokenClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: SudarshanaTokenClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate_val_loss(
    model: SudarshanaTokenClassifier,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def run_training(
    seed: int = 42,
    epochs: int = 3,
    lr: float = 3e-5,
    batch_size: int = 8,
    include_hard_negatives: bool = True,
    backbone: str = "microsoft/deberta-v3-xsmall",
    output_dir: str = "checkpoints/prototype-token-head-deberta-v3-xsmall-20261110"
) -> Dict[str, Any]:
    """Train token head prototype with fixed budget and seed."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Starting training run: seed={seed}, epochs={epochs}, lr={lr}, device={device}")
    print(f"Hard negatives: {include_hard_negatives}, Backbone: {backbone}")
    print(f"=======================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_records, _ = load_dataset_split("train")
    val_records, _ = load_dataset_split("val")

    if not include_hard_negatives:
        train_records = [r for r in train_records if r.get("attack_type") != "hard_negative"]
        print(f"Filtered out hard negatives for ablation: {len(train_records)} records remaining.")

    train_dataset = PromptInjectionDataset(train_records, tokenizer, max_length=256, stride=64)
    val_dataset = PromptInjectionDataset(val_records, tokenizer, max_length=256, stride=64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SudarshanaTokenClassifier(backbone_name_or_path=backbone, num_labels=3)
    model.to(device)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate_val_loss(model, val_loader, device)
        elapsed = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "time_sec": round(elapsed, 2)
        })
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} ({elapsed:.1f}s)")

    total_duration = time.time() - start_time

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Save configuration and run metadata
    config_data = {
        "prototype": True,
        "notice": "This is a prototype checkpoint for head and redaction validation. P3 owns the final joint training run.",
        "backbone": backbone,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "include_hard_negatives": include_hard_negatives,
        "total_duration_sec": round(total_duration, 2),
        "history": history,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"\nSaved prototype checkpoint and metadata to {output_dir}")
    return config_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_hard_negatives", action="store_true")
    parser.add_argument("--output_dir", type=str, default="checkpoints/prototype-token-head-deberta-v3-xsmall-20261110")
    args = parser.parse_args()

    run_training(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        include_hard_negatives=not args.no_hard_negatives,
        output_dir=args.output_dir
    )

