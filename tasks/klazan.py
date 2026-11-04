"""
Klazan, the synthetic dwarf language of dwarfgpt: instruction conversations built from
the packed corpus' translation pairs (parallel/{train,val}.jsonl, {"id", "en", "kz"}).

Three kinds, in the proportions given by `mix`:
- en2kz: the user asks for a paragraph in Klazan, the assistant gives its translation
- kz2en: the same, the other way
- story: the user asks for a short story in Klazan, the assistant tells one

Paragraphs are the unit because the translator keeps paragraphs aligned (sentence
splitting is not reliable inside quotes). The request wording varies, chosen per example
from a fixed list, so the model learns the request rather than one template.

Train examples are read from three quarters of the way into train.jsonl, val examples
from the end of val.jsonl, which keeps them clear of the sentences scripts/klazan_eval.py
tests on (those come from the start of val.jsonl).
"""

import os
import json
import random

from tasks.common import Task

EN2KZ = [
    "Translate into Klazan: {text}",
    "How would you say this in Klazan?\n\n{text}",
    "Can you put this into Klazan for me?\n{text}",
    "Translate this to Klazan.\n\n{text}",
    "What is this in Klazan? \"{text}\"",
    "Please write this in Klazan: {text}",
]
KZ2EN = [
    "Translate from Klazan into English: {text}",
    "What does this Klazan mean?\n\n{text}",
    "Can you translate this Klazan for me?\n{text}",
    "Translate this to English.\n\n{text}",
    "What is this in English? \"{text}\"",
    "Please put this Klazan into English: {text}",
]
STORY = [
    "Tell me a short story in Klazan.",
    "Write a short story in Klazan, please.",
    "Can you tell me a story in Klazan?",
    "I'd like a children's story in Klazan.",
    "Make up a little story in Klazan.",
]


def _read_lines(path, start_frac, count):
    """Up to `count` JSON lines starting at `start_frac` of the file's bytes (wrapping)."""
    size = os.path.getsize(path)
    rows = []
    with open(path, "rb") as f:
        f.seek(int(size * start_frac))
        if start_frac > 0:
            f.readline()                     # skip the partial line we landed in
        while len(rows) < count:
            line = f.readline()
            if not line:
                f.seek(0)
                continue
            rows.append(json.loads(line))
    return rows


class Klazan(Task):

    def __init__(self, corpus_dir, split, num_examples, mix=(0.4, 0.4, 0.2), max_par_chars=600, max_story_chars=1500, seed=42, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "Klazan split must be train|val"
        path = os.path.join(corpus_dir, "parallel", f"{split}.jsonl")
        # read generously: some stories are skipped as too long
        source = _read_lines(path, 0.75 if split == "train" else 0.95, 2 * num_examples)
        rng = random.Random(seed)
        kinds = ["en2kz", "kz2en", "story"]
        self.examples = []
        for row in source:
            if len(self.examples) >= num_examples:
                break
            kind = rng.choices(kinds, weights=mix)[0]
            if kind == "story":
                if len(row["kz"]) > max_story_chars:
                    continue
                user, assistant = rng.choice(STORY), row["kz"].strip()
            else:
                pairs = [(e.strip(), k.strip()) for e, k in zip(row["en"].split("\n"), row["kz"].split("\n"))
                         if e.strip() and k.strip() and len(e) <= max_par_chars]
                if not pairs or row["en"].count("\n") != row["kz"].count("\n"):
                    continue
                en, kz = rng.choice(pairs)
                src, tgt, templates = (en, kz, EN2KZ) if kind == "en2kz" else (kz, en, KZ2EN)
                user, assistant = rng.choice(templates).format(text=src), tgt
            self.examples.append({"kind": kind, "id": row["id"], "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]})
        assert len(self.examples) == num_examples, f"only built {len(self.examples)} of {num_examples} Klazan examples"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        return {"messages": self.examples[index]["messages"]}
