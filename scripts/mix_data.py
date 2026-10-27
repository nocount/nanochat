"""
Build a pretraining data directory that mixes a second language into the base dataset.

Each base train shard is rewritten with documents from --mix-dir interleaved between its
documents, so that the mixed-in documents make up --mix-frac of the tokens, counted with
the trained tokenizer, both within each shard and cumulatively. The mixed-in language
replaces base data rather than adding to it: a run on any mix directory takes the same
number of steps and sees the same number of tokens. If --mix-dir runs out, it starts again
from its first document, and the manifest records how many passes were made.

The validation shard is the base dataset's own (symlinked), so validation bpb measures the
base language alone and stays comparable across mixes. With --mix-frac 0 the train shards
are symlinked too, which gives the control run a directory of the same shape.

Point training at the result with NANOCHAT_DATA_DIR, e.g.:
python -m scripts.mix_data --mix-dir path/to/kz/train --mix-frac 0.10 --num-shards 40 --out mix10
NANOCHAT_DATA_DIR=~/.cache/nanochat/mix10 python -m scripts.base_train ...
"""
import os
import json
import time
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from nanochat.common import get_base_dir
from nanochat.dataset import DOWNLOAD_DIR, MAX_SHARD, index_to_filename
from nanochat.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Mix a second language into the pretraining shards")
parser.add_argument("--mix-dir", type=str, required=True, help="directory of parquet files (a 'text' column) to mix in")
parser.add_argument("--mix-frac", type=float, required=True, help="fraction of tokens taken from --mix-dir, e.g. 0.10")
parser.add_argument("--num-shards", type=int, required=True, help="number of base train shards to use, from shard 0")
parser.add_argument("--out", type=str, required=True, help="output directory name, under the nanochat base dir unless absolute")
parser.add_argument("--row-group-size", type=int, default=1024, help="rows per parquet row group (base shards use 1024)")
args = parser.parse_args()
assert 0 <= args.mix_frac < 1, "--mix-frac must be in [0, 1)"

out_dir = args.out if os.path.isabs(args.out) else os.path.join(get_base_dir(), args.out)
assert not os.path.exists(out_dir), f"{out_dir} already exists; delete it or pick another --out"
base_paths = [os.path.join(DOWNLOAD_DIR, index_to_filename(i)) for i in range(args.num_shards)]
val_path = os.path.join(DOWNLOAD_DIR, index_to_filename(MAX_SHARD))
missing = [p for p in base_paths + [val_path] if not os.path.exists(p)]
assert not missing, f"missing base shards, run `python -m nanochat.dataset -n {args.num_shards}` first: {missing[:3]}"
os.makedirs(out_dir)

tokenizer = get_tokenizer()
def count_tokens(docs):
    return [len(t) for t in tokenizer.encode(docs, num_threads=os.cpu_count())]

def mix_batches():
    """Endless (docs, token_counts) batches from --mix-dir, one row group at a time."""
    paths = sorted(os.path.join(args.mix_dir, f) for f in os.listdir(args.mix_dir) if f.endswith(".parquet"))
    assert paths, f"no parquet files in {args.mix_dir}"
    while True:
        for path in paths:
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                docs = pf.read_row_group(rg_idx).column("text").to_pylist()
                yield docs, count_tokens(docs)
        stats["mix_passes"] += 1

stats = {"base_docs": 0, "base_tokens": 0, "mix_docs": 0, "mix_tokens": 0, "mix_passes": 0, "shards": []}
mix, pending = mix_batches(), []
t0 = time.time()
for i, base_path in enumerate(base_paths):
    out_path = os.path.join(out_dir, os.path.basename(base_path))
    if args.mix_frac == 0:
        os.symlink(base_path, out_path)
        continue
    base_docs = pq.read_table(base_path, columns=["text"]).column("text").to_pylist()
    shard = {"base_docs": len(base_docs), "base_tokens": 0, "mix_docs": 0, "mix_tokens": 0}
    out = []
    for doc, n in zip(base_docs, count_tokens(base_docs)):
        # top up the mixed-in language until it is mix_frac of all tokens so far, this doc included
        total = stats["base_tokens"] + stats["mix_tokens"] + n
        while stats["mix_tokens"] < args.mix_frac * total:
            if not pending:
                docs, counts = next(mix)
                pending = list(zip(docs, counts))[::-1]
            mix_doc, m = pending.pop()
            out.append(mix_doc)
            stats["mix_docs"] += 1; stats["mix_tokens"] += m
            shard["mix_docs"] += 1; shard["mix_tokens"] += m
            total += m
        out.append(doc)
        stats["base_docs"] += 1; stats["base_tokens"] += n
        shard["base_tokens"] += n
    pq.write_table(pa.table({"text": out}), out_path, row_group_size=args.row_group_size, compression="zstd")
    stats["shards"].append(shard)
    frac = stats["mix_tokens"] / (stats["base_tokens"] + stats["mix_tokens"])
    print(f"{os.path.basename(out_path)}: {len(out):,} docs ({shard['mix_docs']:,} mixed in), "
          f"cumulative mix fraction {frac:.4f} | {time.time() - t0:.0f}s")

os.symlink(val_path, os.path.join(out_dir, os.path.basename(val_path)))
manifest = {
    "mix_dir": os.path.abspath(args.mix_dir), "mix_frac": args.mix_frac,
    "base_dir": DOWNLOAD_DIR, "num_shards": args.num_shards, "val_shard": os.path.basename(val_path),
    "tokenizer_vocab_size": tokenizer.get_vocab_size(),
    **{k: v for k, v in stats.items() if k != "shards"}, "per_shard": stats["shards"],
}
# keep the mixed-in corpus's provenance (e.g. a pack_manifest.json one level above --mix-dir)
for cand in (os.path.join(args.mix_dir, "..", "..", "pack_manifest.json"), os.path.join(args.mix_dir, "..", "pack_manifest.json")):
    if os.path.exists(cand):
        with open(cand) as f:
            manifest["mix_source_manifest"] = json.load(f)
        break
with open(os.path.join(out_dir, "mix_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"wrote {out_dir}: {args.num_shards} train shards + val, mix fraction "
      f"{stats['mix_tokens'] / max(1, stats['base_tokens'] + stats['mix_tokens']):.4f}, "
      f"{stats['mix_tokens']:,} mixed-in tokens, {stats['mix_passes']} full passes over --mix-dir")
