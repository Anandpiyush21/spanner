"""Loading QA datasets and turning them into BERT training features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class QAExample:
    question: str
    context: str
    answer: str
    answer_start: int

    @property
    def answer_end(self) -> int:
        return self.answer_start + len(self.answer)


def load_examples(path: str | Path) -> list[QAExample]:
    """Load a dataset of the form ``[{"context": ..., "qas": [{"question", "answer"}, ...]}]``.

    ``answer_start`` may be given explicitly; otherwise the first occurrence of
    the answer in the context is used. Every answer must be an exact span of
    its context, since the model is extractive.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    examples = []
    for record in records:
        context = record["context"]
        for qa in record["qas"]:
            answer = qa["answer"]
            start = qa.get("answer_start", context.find(answer))
            if start < 0 or context[start : start + len(answer)] != answer:
                raise ValueError(
                    f"Answer {answer!r} for question {qa['question']!r} is not a span of its context"
                )
            examples.append(QAExample(qa["question"], context, answer, start))
    return examples


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def make_windows(tokenizer, question: str, context: str, max_length: int = 384, stride: int = 128) -> list[dict]:
    """Split ``[CLS] question [SEP] context [SEP]`` into overlapping windows of ``max_length`` tokens.

    Each window carries ``offsets``: the (start, end) character span in ``context``
    for every context token, and ``None`` for question, special and padding tokens.
    Windowing is done here rather than via ``return_overflowing_tokens`` because
    some ``tokenizers`` releases (0.23.x) emit only a single overflow window for pairs.
    """
    q_ids = tokenizer(question, add_special_tokens=False, verbose=False)["input_ids"][: max_length // 2]
    ctx = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ctx_ids, ctx_offsets = ctx["input_ids"], ctx["offset_mapping"]

    room = max_length - len(q_ids) - 3
    if stride >= room:
        raise ValueError(f"stride ({stride}) must be smaller than the space left for context ({room})")
    with_types = "token_type_ids" in tokenizer.model_input_names

    windows, begin = [], 0
    while True:
        end = min(begin + room, len(ctx_ids))
        input_ids = [tokenizer.cls_token_id, *q_ids, tokenizer.sep_token_id, *ctx_ids[begin:end], tokenizer.sep_token_id]
        offsets = [None] * (len(q_ids) + 2) + [tuple(o) for o in ctx_offsets[begin:end]] + [None]
        n_pad = max_length - len(input_ids)
        window = {
            "input_ids": input_ids + [tokenizer.pad_token_id] * n_pad,
            "attention_mask": [1] * len(input_ids) + [0] * n_pad,
            "offsets": offsets + [None] * n_pad,
        }
        if with_types:
            window["token_type_ids"] = [0] * (len(q_ids) + 2) + [1] * (end - begin + 1) + [0] * n_pad
        windows.append(window)
        if end >= len(ctx_ids):
            return windows
        begin = end - stride


def answer_token_span(offsets: list, answer_start: int, answer_end: int) -> tuple[int, int]:
    """Token indices of the answer within a window, or ``(0, 0)`` if it isn't fully inside."""
    context_tokens = [i for i, o in enumerate(offsets) if o is not None]
    if offsets[context_tokens[0]][0] > answer_start or offsets[context_tokens[-1]][1] < answer_end:
        return 0, 0
    start = next(i for i in context_tokens if offsets[i][1] > answer_start)
    end = next(i for i in reversed(context_tokens) if offsets[i][0] < answer_end)
    return start, end


class QAFeatures(Dataset):
    """Tokenized examples with start/end token labels.

    Long contexts are split into overlapping windows of ``max_length`` tokens.
    Windows that don't contain the answer are labelled with the [CLS] token
    (index 0), the usual SQuAD convention for "no answer here".
    """

    def __init__(self, examples: list[QAExample], tokenizer, max_length: int = 384, stride: int = 128):
        columns: dict[str, list] = {}
        for example in examples:
            for window in make_windows(tokenizer, example.question, example.context, max_length, stride):
                offsets = window.pop("offsets")
                start, end = answer_token_span(offsets, example.answer_start, example.answer_end)
                window["start_positions"], window["end_positions"] = start, end
                for key, value in window.items():
                    columns.setdefault(key, []).append(value)
        self.tensors = {key: torch.tensor(values) for key, values in columns.items()}

    def __len__(self) -> int:
        return len(self.tensors["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.tensors.items()}
