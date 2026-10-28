"""
Build a pretraining data directory that mixes a second language into the base dataset.

Each base train shard is rewritten with documents from --mix-dir interleaved between its
documents, so that the mixed-in documents make up --mix-frac of the tokens, counted with
the trained tokenizer, both within each shard and cumulatively. The mixed-in language
replaces base data rather than adding to it: a run on any mix directory takes the same
number of steps and sees the same number of tokens. If --mix-dir runs out, it starts again
from its first document, and the manifest records how many passes were made.

With --parallel (a JSONL file of {"id", "en", "kz"} translation pairs) and --parallel-frac,
that share of the mixed-in tokens are bilingual documents instead, which alternate the
two languages paragraph by paragraph:
    English: <paragraph 1>
    Klazan: <paragraph 1, translated>
    English: <paragraph 2>
    ...
Half of them (chosen by a hash of the id) put the second language first, so both
directions are seen. Pairs are read from the middle of the file onwards, wrapping around,
so that a story is not seen monolingually and as a pair at the same point in training.

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
import zlib
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
parser.add_argument("--parallel", type=str, default=None, help="JSONL of {id, en, kz} translation pairs for bilingual documents")
parser.add_argument("--parallel-frac", type=float, default=0.0, help="share of the mixed-in tokens that are bilingual documents (default: 0)")
parser.add_argument("--labels", type=str, default="English,Klazan", help="line labels for the two languages in bilingual documents")
parser.add_argument("--row-group-size", type=int, default=1024, help="rows per parquet row group (base shards use 1024)")
args = parser.parse_args()
assert 0 <= args.mix_frac < 1, "--mix-frac must be in [0, 1)"
assert 0 <= args.parallel_frac <= 1, "--parallel-frac must be in [0, 1]"
assert (args.parallel_frac > 0) == (args.parallel is not None), "--parallel and --parallel-frac go together"
labels = args.labels.split(",")

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

def bilingual_doc(pair):
    """Alternate the two languages paragraph by paragraph; the order is fixed per id."""
    first, second = ("en", "kz") if zlib.crc32(pair["id"].encode()) % 2 == 0 else ("kz", "en")
    name = {"en": labels[0], "kz": labels[1]}
    a, b = pair[first].split("\n"), pair[second].split("\n")
    assert len(a) == len(b), f"paragraph counts differ for {pair['id']}"
    lines = []
    for pa_, pb_ in zip(a, b):
        if pa_.strip() or pb_.strip():
            lines += [f"{name[first]}: {pa_.strip()}", f"{name[second]}: {pb_.strip()}"]
    return "\n".join(lines)

def parallel_batches(batch_size=1024):
    """Endless (docs, token_counts) batches of bilingual documents, from the middle of the file."""
    size = os.path.getsize(args.parallel)
    with open(args.parallel, "rb") as f:
        f.seek(size // 2)
        f.readline()                         # skip the partial line we landed in
        stats["parallel_start_byte"] = f.tell()
        batch = []
        while True:
            line = f.readline()
            if not line:                     # wrap around to the start of the file
                if f.tell() == 0:
                    raise ValueError(f"{args.parallel} is empty")
                f.seek(0)
                stats["parallel_wraps"] += 1
                continue
            batch.append(bilingual_doc(json.loads(line)))
            if len(batch) == batch_size:
                yield batch, count_tokens(batch)
                batch = []

stats = {"base_docs": 0, "base_tokens": 0, "mix_docs": 0, "mix_tokens": 0, "mix_passes": 0,
         "parallel_docs": 0, "parallel_tokens": 0, "parallel_wraps": 0, "shards": []}
sources = {"mono": [mix_batches(), []]}
if args.parallel:
    sources["parallel"] = [parallel_batches(), []]

def next_mix_doc():
    """The next mixed-in document: bilingual while they are under their share, else monolingual."""
    use_parallel = args.parallel and stats["parallel_tokens"] < args.parallel_frac * stats["mix_tokens"]
    gen, pending = sources["parallel" if use_parallel else "mono"]
    if not pending:
        docs, counts = next(gen)
        pending.extend(reversed(list(zip(docs, counts))))
    doc, n = pending.pop()
    if use_parallel:
        stats["parallel_docs"] += 1; stats["parallel_tokens"] += n
    return doc, n

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
            mix_doc, m = next_mix_doc()
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
    "parallel": os.path.abspath(args.parallel) if args.parallel else None,
    "parallel_frac": args.parallel_frac, "labels": labels,
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
total = max(1, stats["base_tokens"] + stats["mix_tokens"])
print(f"wrote {out_dir}: {args.num_shards} train shards + val, mix fraction {stats['mix_tokens'] / total:.4f} "
      f"({stats['parallel_tokens'] / max(1, stats['mix_tokens']):.4f} of it bilingual), "
      f"{stats['mix_tokens']:,} mixed-in tokens, {stats['mix_passes']} full passes over --mix-dir")
