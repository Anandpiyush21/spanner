"""SQuAD-style Exact Match and F1 scoring."""

from __future__ import annotations

import re
import string
from collections import Counter

from .data import QAExample
from .predict import QuestionAnswerer


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, truth: str) -> float:
    return float(normalize(prediction) == normalize(truth))


def f1(prediction: str, truth: str) -> float:
    pred_tokens, truth_tokens = normalize(prediction).split(), normalize(truth).split()
    common = sum((Counter(pred_tokens) & Counter(truth_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(qa: QuestionAnswerer, examples: list[QAExample], verbose: bool = False) -> dict[str, float]:
    em_total = f1_total = 0.0
    for example in examples:
        prediction = qa.answer(example.question, example.context).text
        em, score = exact_match(prediction, example.answer), f1(prediction, example.answer)
        em_total += em
        f1_total += score
        if verbose:
            mark = "✓" if em else "✗"
            print(f"{mark} {example.question}\n    predicted: {prediction!r}\n    expected:  {example.answer!r}")
    n = len(examples)
    return {"exact_match": 100 * em_total / n, "f1": 100 * f1_total / n, "count": n}
