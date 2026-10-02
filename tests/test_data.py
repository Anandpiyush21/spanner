import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from spanner.data import QAFeatures, load_examples
from spanner.evaluate import exact_match, f1

DATA = Path(__file__).resolve().parent.parent / "data" / "kangaroo_qa.json"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_every_answer_is_a_span_of_its_context():
    examples = load_examples(DATA)
    assert examples
    for e in examples:
        assert e.context[e.answer_start : e.answer_end] == e.answer


def test_answers_outside_context_are_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"context": "Kangaroos hop.", "qas": [{"question": "Do they fly?", "answer": "no"}]}]))
    with pytest.raises(ValueError):
        load_examples(bad)


def test_labels_point_at_the_answer_tokens(tokenizer):
    examples = load_examples(DATA)
    features = QAFeatures(examples, tokenizer)
    assert len(features) == len(examples)
    for i, e in enumerate(examples):
        item = features[i]
        span = item["input_ids"][item["start_positions"] : item["end_positions"] + 1]
        assert tokenizer.decode(span).replace(" ", "") == e.answer.lower().replace(" ", "")


def test_long_contexts_are_windowed(tokenizer, tmp_path):
    context = "Kangaroos hop across the outback. " * 120 + "The answer is forty two."
    path = tmp_path / "long.json"
    path.write_text(json.dumps([{"context": context, "qas": [{"question": "What is the answer?", "answer": "forty two"}]}]))
    features = QAFeatures(load_examples(path), tokenizer, max_length=128, stride=32)
    assert len(features) > 1
    labelled = [features[i] for i in range(len(features)) if features[i]["start_positions"] > 0]
    assert labelled
    span = labelled[-1]["input_ids"][labelled[-1]["start_positions"] : labelled[-1]["end_positions"] + 1]
    assert tokenizer.decode(span) == "forty two"


def test_metrics():
    assert exact_match("The red kangaroo", "red kangaroo") == 1.0
    assert f1("grasses and vegetation", "grasses and other vegetation") == pytest.approx(2 * 1 * 0.75 / 1.75)
    assert f1("joeys", "mobs") == 0.0
