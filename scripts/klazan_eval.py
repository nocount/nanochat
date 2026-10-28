"""
Evaluate a base checkpoint on Klazan, the second language of the dwarfgpt mixes,
and write a JSON report.

1. bpb on three held-out sets: Klazan stories, the same stories in English, and
   ClimbMix val (English web text, the base dataset's own validation shard), computed
   the way base_train computes val bpb.
2. Translation, EN->KZ and KZ->EN, on held-out single sentences (one per story, never
   the opening one), in the line format of
   the bilingual training documents ("English: ...\\nKlazan: ..."). Zero-shot and with
   --shots example pairs before it. Greedy decoding; scored by exact match and chrF
   against the reference, which is the deterministic translator's output.
3. Free continuations of Klazan story openings. With --klazan-repo, they are checked by
   that repo's validator (klazan.validate), which reports unknown words and loans.

python -m scripts.klazan_eval --model-tag d12-mix10p --corpus path/to/packed --out report.json
"""
import os
import re
import json
import time
import random
import argparse
import tempfile
import subprocess
from collections import Counter
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import nanochat.dataset
from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.dataset import DOWNLOAD_DIR
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

parser = argparse.ArgumentParser(description="Klazan evaluation of a base checkpoint")
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: last)")
parser.add_argument("--corpus", type=str, required=True, help="packed Klazan corpus (has kz/val and parallel/val.jsonl)")
parser.add_argument("--out", type=str, default=None, help="report path (default: <base dir>/logs/<model tag>-klazan_eval.json)")
parser.add_argument("--bpb-tokens", type=int, default=80*524288, help="tokens for ClimbMix val bpb (default: base_train's --eval-tokens)")
parser.add_argument("--story-bpb-tokens", type=int, default=16*524288, help="tokens for each story set (they hold ~19M, so no repeats)")
parser.add_argument("--num-sentences", type=int, default=200, help="translation test sentences per direction")
parser.add_argument("--shots", type=int, default=4, help="example pairs for the few-shot translation run")
parser.add_argument("--klazan-repo", type=str, default=None, help="klazan checkout, to validate generated text")
parser.add_argument("--labels", type=str, default="English,Klazan")
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()
EN, KZ = args.labels.split(",")

_, _, _, _, device = compute_init(autodetect_device_type())
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)
bos = tokenizer.get_bos_token_id()
T = meta["model_config"]["sequence_len"]
report = {"model_tag": args.model_tag, "step": meta.get("step"), "corpus": os.path.abspath(args.corpus)}
t0 = time.time()

# -----------------------------------------------------------------------------
# 1. bpb, computed exactly as base_train computes val bpb: nanochat's BOS-aligned best-fit
#    loader over the last parquet file of a data directory. The Klazan and English story
#    sets get small directories of their own (built once, under the nanochat base dir).

def bpb_of(data_dir, tokens):
    nanochat.dataset.DATA_DIR = data_dir     # list_parquet_files() reads it at call time
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.batch_size, T, "val", device=device)
    steps = tokens // (args.batch_size * T)
    return evaluate_bpb(model, loader, steps, get_token_bytes(device=device))

def eval_dir(name, docs):
    """A data directory whose val split (its last file) holds docs."""
    d = os.path.join(get_base_dir(), name)
    path = os.path.join(d, "shard_00001.parquet")
    if not os.path.exists(path):
        os.makedirs(d, exist_ok=True)
        pq.write_table(pa.table({"text": docs}), path + ".tmp", row_group_size=1024, compression="zstd")
        os.replace(path + ".tmp", path)
    return d

kz_val_path = sorted(os.path.join(args.corpus, "kz/val", f) for f in os.listdir(os.path.join(args.corpus, "kz/val")))[0]
kz_docs = pq.read_table(kz_val_path).column("text").to_pylist()
kz_set = set(kz_docs)
parallel_val = [json.loads(l) for l in open(os.path.join(args.corpus, "parallel/val.jsonl"))]
en_docs = [r["en"] for r in parallel_val if r["kz"] in kz_set]  # the same stories, in English
report["bpb"] = {
    "klazan_val": bpb_of(eval_dir("eval_klazan_val", kz_docs), args.story_bpb_tokens),
    "tinystories_en_val": bpb_of(eval_dir("eval_tinystories_en_val", en_docs), args.story_bpb_tokens),
    "climbmix_val": bpb_of(DOWNLOAD_DIR, args.bpb_tokens),   # its last file is the ClimbMix val shard
}
report["bpb_tokens"] = {"stories": args.story_bpb_tokens, "climbmix": args.bpb_tokens}
print0(f"bpb: {report['bpb']} | {time.time() - t0:.0f}s")

# -----------------------------------------------------------------------------
# 2. translation of held-out sentences

def generate(prompt, max_tokens, temperature=0.0, n=1, seed=42):
    ids = [bos] + tokenizer.encode(prompt)
    outs, _ = engine.generate_batch(ids, num_samples=n, max_tokens=max_tokens, temperature=temperature, seed=seed)
    return [tokenizer.decode(o[len(ids):]) for o in outs]

def chrf(hyp, ref, n_max=6, beta=2.0):
    """chrF (Popović 2015): character n-gram F-score, n = 1..6, recall weighted by beta."""
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    precisions, recalls = [], []
    for n in range(1, n_max + 1):
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        match = sum((h & r).values())
        precisions.append(match / sum(h.values())); recalls.append(match / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)

split_sentences = lambda t: re.split(r"(?<=[.!?])\s+", t.strip())
pairs = []
for r in parallel_val:
    if r["kz"] in kz_set:
        continue  # keep the bpb stories and the translation sentences apart
    e, k = split_sentences(r["en"]), split_sentences(r["kz"])
    if len(e) != len(k) or len(e) < 2:
        continue
    # one sentence per story, never the first: openings are formulaic ("Once upon a time"),
    # so a model that ignores the source still scores well on them
    i = random.Random(r["id"]).randrange(1, len(e))
    # skip sentences that cross a paragraph break or hold a quote (the split goes wrong inside quotes)
    if not re.search(r'[\n"“”]', e[i] + k[i]) and 5 <= len(e[i].split()) <= 20:
        pairs.append({"en": e[i], "kz": k[i]})
    if len(pairs) >= args.shots + args.num_sentences:
        break
shots, tests = pairs[:args.shots], pairs[args.shots:]
report["translation"] = {}
for direction, (src, tgt) in (("en2kz", ("en", "kz")), ("kz2en", ("kz", "en"))):
    label = {"en": EN, "kz": KZ}
    for k in (0, args.shots):
        prefix = "".join(f"{label[src]}: {p[src]}\n{label[tgt]}: {p[tgt]}\n" for p in shots[:k])
        rows = []
        for p in tests:
            cap = 3 * len(tokenizer.encode(p[src])) + 8  # from the source only: the engine has no stop string
            hyp = generate(prefix + f"{label[src]}: {p[src]}\n{label[tgt]}:", max_tokens=cap)[0]
            hyp = hyp.split("\n")[0].strip()
            rows.append({"src": p[src], "ref": p[tgt], "hyp": hyp, "exact": hyp == p[tgt], "chrf": chrf(hyp, p[tgt])})
        key = f"{direction}_{k}shot"
        report["translation"][key] = {
            "exact": sum(r["exact"] for r in rows) / len(rows),
            "chrf": sum(r["chrf"] for r in rows) / len(rows),
            "n": len(rows), "examples": rows[:10],
        }
        print0(f"{key}: exact {report['translation'][key]['exact']:.3f}, chrF {report['translation'][key]['chrf']:.3f} | {time.time() - t0:.0f}s")

# -----------------------------------------------------------------------------
# 3. free continuations of Klazan story openings

openings = ["Gez drur zengi, zad un", "Un zeng, Lily trukad dra ta", "Ta dwergaz zad grak"]
continuations = [o + s for o in openings for s in generate(o, max_tokens=128, temperature=0.8, n=4)]
report["continuations"] = continuations
if args.klazan_repo:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for c in continuations:
            f.write(json.dumps({"id": "gen", "en": "", "kz": c}) + "\n")
    out = subprocess.run(["uv", "run", "--quiet", "python", "-m", "klazan.validate", f.name],
                         cwd=args.klazan_repo, capture_output=True, text=True)
    os.unlink(f.name)
    report["continuation_validation"] = (out.stdout + out.stderr).strip().splitlines()[-3:]
    print0("\n".join(report["continuation_validation"]))

out_path = args.out or os.path.join(get_base_dir(), "logs", f"{args.model_tag}-klazan_eval.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print0(f"wrote {out_path} | {time.time() - t0:.0f}s")
