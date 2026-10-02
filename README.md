# Spanner

**Ask questions about your own documents with a fine-tuned BERT model.**

**Spanner** is a small, readable toolkit for *extractive* question answering. Give it a passage of text and a question, and it highlights the span of the passage that answers it. It ships with a kangaroo knowledge set as a worked example, but you can point it at any text and train it on your own question/answer pairs.

```console
$ spanner ask "What do kangaroos eat?"
Answer: grasses and other vegetation  (confidence 0.98)

$ spanner ask "Where do female kangaroos nurture their young?"
Answer: in a specialized pouch  (confidence 0.98)
```

---

## Features

- **One command each** to train, evaluate and ask: `spanner train`, `spanner eval`, `spanner ask`.
- **Correct span labels.** Character-level answer positions are mapped to token positions with the tokenizer's offset mapping, so the model learns exactly which tokens make up the answer.
- **Long documents.** Contexts longer than the model's window are split into overlapping chunks, and the best answer across all chunks is returned.
- **Strong starting point.** Fine-tuning starts from a BERT model already trained on SQuAD 2.0 ([`deepset/bert-base-uncased-squad2`](https://huggingface.co/deepset/bert-base-uncased-squad2)), so a handful of domain examples goes a long way. Any Hugging Face QA-compatible model can be used instead.
- **SQuAD-style metrics.** Exact Match and token-level F1.
- **Runs anywhere.** Picks CUDA, Apple Silicon (MPS) or CPU automatically.

## How it works

```
 question ─┐
           ├─► [CLS] question [SEP] context window [SEP] ─► BERT ─► start / end logits
 context ──┘        (sliding windows, stride 128)                        │
                                                                         ▼
                                     best span: argmax P(start) · P(end), start ≤ end
                                                                         │
                                                                         ▼
                                                      answer text sliced from the context
```

BERT reads the question and a window of the context together and predicts, for every context token, how likely it is to be the first or last token of the answer. The answer is the span with the highest combined probability.

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/Anandpiyush21/NLP_Project.git
cd NLP_Project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # add ".[dev]" for pytest and Jupyter
```

> **CPU only?** Install the smaller CPU build of PyTorch first:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

Then:

```bash
spanner train      # fine-tune on data/kangaroo_qa.json  →  models/spanner-kangaroo/
spanner eval -v    # Exact Match / F1, with every prediction printed
spanner ask        # interactive Q&A over data/kangaroo.txt
```

On a CPU, training on the bundled data takes around a minute and a half.

## Usage

### Ask questions

```bash
# one-off question
spanner ask "How fast can a red kangaroo run?"

# interactive session over any text file
spanner ask --context path/to/document.txt

# no training needed: use any QA model from the Hugging Face Hub
spanner ask --model deepset/bert-base-uncased-squad2 "What are young kangaroos called?"
```

### Train

```bash
spanner train \
  --data data/kangaroo_qa.json \
  --output models/spanner-kangaroo \
  --base-model deepset/bert-base-uncased-squad2 \
  --epochs 3 --batch-size 8 --lr 3e-5
```

| Option | Default | Description |
| --- | --- | --- |
| `--data` | `data/kangaroo_qa.json` | Training set (format below) |
| `--output` | `models/spanner-kangaroo` | Where the model and tokenizer are saved |
| `--base-model` | `deepset/bert-base-uncased-squad2` | Starting checkpoint. Use `bert-base-uncased` to train from plain BERT |
| `--epochs` | `3` | Passes over the data |
| `--batch-size` | `8` | Examples per step |
| `--lr` | `3e-5` | Peak learning rate (AdamW, linear warmup and decay) |
| `--max-length` | `384` | Tokens per window |
| `--stride` | `128` | Overlap between consecutive windows |
| `--seed` | `42` | Random seed |
| `--device` | auto | `cpu`, `cuda` or `mps` (a global flag: `spanner --device cpu train`) |

### Evaluate

```bash
spanner eval --data data/kangaroo_qa.json --model models/spanner-kangaroo -v
```

### From Python

```python
from spanner.predict import QuestionAnswerer

qa = QuestionAnswerer("models/spanner-kangaroo")
answer = qa.answer("What is a group of kangaroos called?", open("data/kangaroo.txt").read())
print(answer.text, answer.score, answer.start, answer.end)
```

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) runs through the whole pipeline step by step.

## Using your own data

Training data is a JSON list of passages, each with its questions. **Every answer must appear word-for-word in its passage.** The loader checks this and raises an error if one doesn't.

```json
[
  {
    "context": "Kangaroos are social animals and often form groups known as mobs.",
    "qas": [
      { "question": "What is a group of kangaroos called?", "answer": "mobs" }
    ]
  }
]
```

If an answer appears more than once in the passage, add `"answer_start": <character index>` to say which occurrence you mean.

## Results

On the bundled kangaroo set (17 questions over 5 passages):

| Model | Exact Match | F1 |
| --- | --- | --- |
| `deepset/bert-base-uncased-squad2`, no fine-tuning | 82.4 | 97.1 |
| + 3 epochs on the kangaroo set | 100.0 | 100.0 |

These numbers are measured on the training questions, so they show that the model fits the domain, not how well it generalises. To measure that, write a held-out file in the same format and pass it with `spanner eval --data`.

## Project structure

```
.
├── data/
│   ├── kangaroo.txt          # document to ask questions about
│   └── kangaroo_qa.json      # question/answer training set
├── notebooks/
│   └── walkthrough.ipynb     # end-to-end demo
├── src/spanner/
│   ├── cli.py                # `spanner` command
│   ├── data.py               # loading, sliding windows, span labels
│   ├── train.py              # fine-tuning loop
│   ├── predict.py            # QuestionAnswerer
│   ├── evaluate.py           # Exact Match / F1
│   └── utils.py              # device selection, seeding
├── tests/
│   └── test_data.py
└── pyproject.toml
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The tests check that every training answer maps back to the right tokens, that long contexts are windowed correctly, and that the metrics are computed correctly.

## Authors

- **Piyush** ([@Anandpiyush21](https://github.com/Anandpiyush21))
- **Nirdesh Gothania** ([@NirdeshGothania](https://github.com/NirdeshGothania))
