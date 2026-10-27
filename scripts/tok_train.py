"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--mix-dir', type=str, default=None, help='Directory of parquet files (a "text" column) to mix in, e.g. a second language (default: none)')
parser.add_argument('--mix-frac', type=float, default=0.15, help='Fraction of training characters taken from --mix-dir (default: 0.15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
if args.mix_dir:
    print(f"mix_dir: {args.mix_dir} (mix_frac: {args.mix_frac})")

# -----------------------------------------------------------------------------
# Text iterator

def mix_docs():
    """Documents from the --mix-dir parquet files, in file and row group order."""
    paths = sorted(os.path.join(args.mix_dir, f) for f in os.listdir(args.mix_dir) if f.endswith('.parquet'))
    assert paths, f"no parquet files in {args.mix_dir}"
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            yield from pf.read_row_group(rg_idx).column('text').to_pylist()

mix_stats = {"chars": 0, "docs": 0}
def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) If --mix-dir is set, interleave its documents so that they make up
       args.mix_frac of the characters seen so far
    3) Crop every document to args.doc_cap characters
    4) Break when we've seen args.max_chars characters
    """
    nchars = 0
    mix = mix_docs() if args.mix_dir else None
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            docs, pending = [], 0
            while mix is not None and mix_stats["chars"] < args.mix_frac * (nchars + pending + len(doc)):
                mix_doc = next(mix, None)
                if mix_doc is None:
                    print(f"WARNING: --mix-dir ran out after {mix_stats['chars']:,} characters")
                    mix = None
                    break
                mix_doc = mix_doc[:args.doc_cap]
                docs.append(mix_doc)
                mix_stats["chars"] += len(mix_doc)
                mix_stats["docs"] += 1
                pending += len(mix_doc)
            docs.append(doc)
            for doc_text in docs:
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")
if args.mix_dir:
    print(f"Mixed in {mix_stats['docs']:,} documents, {mix_stats['chars']:,} characters from {args.mix_dir}")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0) # special tokens are not counted
    else:
        # use the raw bytes of the token: decoding to a string first corrupts
        # tokens that are not valid standalone UTF-8 (e.g. the raw bytes >= 0x80)
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
