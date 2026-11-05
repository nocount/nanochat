"""
Klazan, the synthetic dwarf language of dwarfgpt: instruction conversations built from
the packed corpus' translation pairs (parallel/{train,val}.jsonl, {"id", "en", "kz"}).

Four kinds, in the proportions given by `mix`:
- en2kz:  the user asks for a sentence or paragraph in Klazan, the assistant translates it
- kz2en:  the same, the other way
- story:  the user asks for a short story in Klazan, the assistant tells one
- story2: the same, then the user asks what it means and the assistant gives the English

Half the translations are single sentences, half whole paragraphs. Paragraphs are the
translator's aligned unit; a sentence is used only where both sides split into the same
number of sentences and there is no quote to split inside.

Requests are composed from parts (a verb phrase, an object, the target language, a
layout), which gives hundreds of distinct wordings, so the model learns to recognize the
request and not one template. Wordings that scripts/klazan_eval.py tests chat models
with are held out: no verb phrase starts "Could you", and story requests never mention
bedtime, dragons or two friends.

Train examples are read from three quarters of the way into train.jsonl, val examples
from the end of val.jsonl, which keeps them clear of the sentences klazan_eval.py tests
on (those come from the start of val.jsonl).
"""

import os
import re
import json
import random

from tasks.common import Task

# --- request wording, composed from parts --------------------------------------------
# {obj} is what is being translated, {lang} the target language.
ASK_VERBS = [
    "Translate {obj} into {lang}", "Translate {obj} to {lang}", "Please translate {obj} into {lang}",
    "Can you translate {obj} into {lang}", "Would you translate {obj} into {lang}",
    "I need {obj} in {lang}", "Put {obj} into {lang}", "Please put {obj} into {lang}",
    "Turn {obj} into {lang}", "Render {obj} in {lang}", "Write {obj} in {lang}",
    "How do you say {obj} in {lang}", "How would you say {obj} in {lang}",
    "What is {obj} in {lang}", "Give me {obj} in {lang}", "Help me translate {obj} into {lang}",
]
OBJECTS = {"sentence": ["this", "this sentence", "the following", "this line", "the sentence below"],
           "paragraph": ["this", "this paragraph", "the following", "this passage", "the text below"]}
QUESTION_STARTS = ("Can ", "Would ", "How ", "What ")
EN2KZ_LANG, KZ2EN_LANG = "Klazan", "English"
KZ2EN_EXTRA = ["What does this Klazan mean?", "What does this mean in English?", "I found this Klazan text. What does it say?",
               "Can you tell me what this Klazan says?"]


def compose_ask(rng, unit, lang, src_lang):
    """One request for a translation of `unit` into `lang`, laid out with {text}."""
    obj = rng.choice(OBJECTS[unit])
    if rng.random() < 0.3:                   # name the source language: "this Klazan sentence"
        obj = re.sub(r"^(this|the)( |$)", rf"\1 {src_lang}\2", obj, count=1).replace(f"{src_lang} following", f"following {src_lang}")
    verb = rng.choice(ASK_VERBS).format(obj=obj, lang=lang)
    end = "?" if verb.startswith(QUESTION_STARTS) else rng.choice([".", ":", ""])
    if rng.random() < 0.1:
        verb, end = verb[0].lower() + verb[1:], end
    layouts = [f"{verb}{end if end != '' else ':'} {{text}}", f"{verb}{end}\n\n{{text}}", f"{verb}{end}\n{{text}}",
               f"{{text}}\n\n{verb}{end or '.'}", f"{verb}{end if end == '?' else ':'} \"{{text}}\""]
    return rng.choice(layouts)

STORY_ASKS = [
    "Tell me a short story in Klazan.", "Write a short story in Klazan, please.", "Can you tell me a story in Klazan?",
    "I'd like a children's story in Klazan.", "Make up a little story in Klazan.", "Tell me a story in Klazan.",
    "Write me a Klazan story.", "Please write a story in Klazan.", "I want to read a short story in Klazan.",
    "Tell me a little tale in Klazan.", "Can you write a simple story in Klazan for me?", "Share a short Klazan story.",
]
STORY_FOLLOWUPS = ["What does that mean in English?", "Can you translate that into English?", "Now tell it in English, please.",
                   "What does it say in English?", "Translate your story into English."]


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

_split_sentences = lambda t: re.split(r"(?<=[.!?])\s+", t.strip())


class Klazan(Task):

    def __init__(self, corpus_dir, split, num_examples, mix=(0.45, 0.35, 0.15, 0.05), sentence_frac=0.5,
                 max_par_chars=600, max_story_chars=1500, seed=42, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "Klazan split must be train|val"
        path = os.path.join(corpus_dir, "parallel", f"{split}.jsonl")
        # read generously: some stories are skipped as too long
        source = _read_lines(path, 0.75 if split == "train" else 0.95, 2 * num_examples)
        rng = random.Random(seed)
        kinds = ["en2kz", "kz2en", "story", "story2"]
        self.examples = []
        for row in source:
            if len(self.examples) >= num_examples:
                break
            kind = rng.choices(kinds, weights=mix)[0]
            if kind in ("story", "story2"):
                if len(row["kz"]) > max_story_chars:
                    continue
                messages = [{"role": "user", "content": rng.choice(STORY_ASKS)},
                            {"role": "assistant", "content": row["kz"].strip()}]
                if kind == "story2":
                    messages += [{"role": "user", "content": rng.choice(STORY_FOLLOWUPS)},
                                 {"role": "assistant", "content": row["en"].strip()}]
            else:
                if row["en"].count("\n") != row["kz"].count("\n"):
                    continue
                pairs = [(e.strip(), k.strip()) for e, k in zip(row["en"].split("\n"), row["kz"].split("\n"))
                         if e.strip() and k.strip() and len(e) <= max_par_chars]
                if not pairs:
                    continue
                en, kz = rng.choice(pairs)
                unit = "paragraph"
                if rng.random() < sentence_frac:
                    es, ks = _split_sentences(en), _split_sentences(kz)
                    if len(es) == len(ks) and not re.search(r'["“”]', en + kz):
                        i = rng.randrange(len(es))
                        en, kz, unit = es[i], ks[i], "sentence"
                if kind == "en2kz":
                    src, tgt = en, kz
                    ask = compose_ask(rng, unit, EN2KZ_LANG, "English")
                else:
                    src, tgt = kz, en
                    ask = rng.choice(KZ2EN_EXTRA) + "\n\n{text}" if rng.random() < 0.15 else compose_ask(rng, unit, KZ2EN_LANG, "Klazan")
                messages = [{"role": "user", "content": ask.replace("{text}", src)}, {"role": "assistant", "content": tgt}]
            self.examples.append({"kind": kind, "id": row["id"], "messages": messages})
        assert len(self.examples) == num_examples, f"only built {len(self.examples)} of {num_examples} Klazan examples"

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        return {"messages": self.examples[index]["messages"]}
