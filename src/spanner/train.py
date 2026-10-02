"""Fine-tuning a BERT model for extractive question answering."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

from .data import QAFeatures, load_examples
from .utils import pick_device, set_seed

DEFAULT_BASE_MODEL = "deepset/bert-base-uncased-squad2"


def train(
    data_path: str | Path,
    output_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 384,
    stride: int = 128,
    seed: int = 42,
    device: str | None = None,
) -> Path:
    set_seed(seed)
    device = pick_device(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(base_model).to(device)

    examples = load_examples(data_path)
    features = QAFeatures(examples, tokenizer, max_length=max_length, stride=stride)
    loader = DataLoader(features, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(examples)} examples ({len(features)} features) from {data_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch}/{epochs}: mean loss {running_loss / len(loader):.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")
    return output_dir
