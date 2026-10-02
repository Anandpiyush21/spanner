"""Answering questions with a (fine-tuned) extractive QA model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from .data import make_windows
from .utils import pick_device


@dataclass
class Answer:
    text: str
    score: float
    start: int
    end: int


class QuestionAnswerer:
    def __init__(self, model_path: str, device: str | None = None, max_length: int = 384, stride: int = 128):
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device).eval()
        self.max_length = max_length
        self.stride = stride

    @torch.no_grad()
    def answer(self, question: str, context: str, max_answer_tokens: int = 30) -> Answer:
        """Return the highest-scoring answer span across all windows of ``context``."""
        windows = make_windows(self.tokenizer, question, context, self.max_length, self.stride)
        all_offsets = [w.pop("offsets") for w in windows]
        batch = {key: torch.tensor([w[key] for w in windows], device=self.device) for key in windows[0]}
        outputs = self.model(**batch)

        best = Answer(text="", score=float("-inf"), start=0, end=0)
        for i, offsets in enumerate(all_offsets):
            is_context = torch.tensor([o is not None for o in offsets])
            start_probs = outputs.start_logits[i].cpu().masked_fill(~is_context, float("-inf")).softmax(-1)
            end_probs = outputs.end_logits[i].cpu().masked_fill(~is_context, float("-inf")).softmax(-1)

            # score[s, e] = P(start=s) * P(end=e), restricted to s <= e < s + max_answer_tokens
            scores = start_probs[:, None] * end_probs[None, :]
            scores = torch.triu(scores) - torch.triu(scores, diagonal=max_answer_tokens)
            s, e = divmod(int(scores.argmax()), scores.shape[1])
            score = float(scores[s, e])

            if score > best.score:
                char_start, char_end = offsets[s][0], offsets[e][1]
                best = Answer(context[char_start:char_end], score, char_start, char_end)
        return best
