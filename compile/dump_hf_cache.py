"""Dump Hugging Face's own KV cache as the feed the compiled ERNIE graph expects, so one decode step
can be checked against the reference without the harness having to build the cache itself.

Why this exists. The chained run (host_gen.py feeding one token per step from an all-zero cache) agrees
with the HF golden at only 1 of 7 teacher-forced prompt positions, mean cos 0.785, and step 0 is
anticorrelated (cos -0.255). That says nothing yet about the compiled net: the same run also pins the
absolute position one slot too high and leaves slot 0 as an all-zero key that nothing masks out, so a
diluted softmax is baked into every step. Seeding the cache from HF and feeding only the last token
removes the chaining and the pad-slot history from the comparison, so whatever error is left belongs to
the graph or the compiler.

Head geometry: the graph was rewritten by tile_kv_heads.py so each of ERNIE's 2 KV heads is replicated
4x, ordered [0,0,0,0,1,1,1,1] (Concat along the width axis, then Split into 8 heads of 128). The fed
cache has to use the same order, so HF's [1, 2, seq, 128] becomes np.repeat(..., 4, axis=1) -> [1, 8, seq, 128].

Slot conventions tested (see --slot-base):
  1  slot 0 stays an all-zero pad and position t lands in slot t+1 -- what the vendored driver does,
     and what host_gen.py's pos-offset 1 chaining assumes.
  0  position t lands in slot t -- what HF's own cache looks like, no pad.

Usage (school): python dump_hf_cache.py --n 6 --slot-base 1 --out D:\\yzd\\WCIS\\MOD\\ernie\\seed6_a
Writes <out>/in_v<id>.ftmp for every past input of the graph (v_id 2..37, fp16, [1, 8, 512, 128]) plus
meta.json with the token ids and the last-position HF logits (step<i>_gold.npy convention).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROMPT = [94515, 94036, 31282, 5502, 44284, 3458, 93977]
MAXSLOT = 512

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=r'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT')
ap.add_argument('--n', type=int, default=6, help='how many prompt tokens go into the cache')
ap.add_argument('--slot-base', type=int, choices=[0, 1], default=1)
ap.add_argument('--out', required=True)
ap.add_argument('--dtype', default='float32', choices=['float32', 'float16'])
ap.add_argument('--rep', type=int, default=4,
                help='how many times each KV head is tiled: 4 matches tile_kv_heads.py on this model, '
                     '1 seeds the native 2-head export')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.src)
dt = {'float32': torch.float32, 'float16': torch.float16}[args.dtype]
model = AutoModelForCausalLM.from_pretrained(args.src, dtype=dt).eval()

ids = torch.tensor([PROMPT[:args.n]])
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
pkv = out.past_key_values
# transformers 4.5x hands back either a Cache object or the legacy tuple of per-layer pairs
layers = []
if hasattr(pkv, 'layers'):
    for lyr in pkv.layers:
        layers.append((lyr.keys, lyr.values))
else:
    for k, v in zip(pkv[0], pkv[1]):
        layers.append((k, v))
print('LAYERS %d shape %s heads %d dim %d seq %d'
      % (len(layers), tuple(layers[0][0].shape), layers[0][0].shape[1],
         layers[0][0].shape[-1], layers[0][0].shape[2]), flush=True)

with torch.no_grad():
    full = model(input_ids=torch.tensor([PROMPT[:args.n + 1]]), use_cache=False)
gold = full.logits[0, -1].float().numpy()

n_written = 0
# the graph declares the cache keys-first, values-last -- past_key_values.0.key .. .17.key, then
# .0.value .. .17.value -- and the compiled v_id is just that declared index with input_ids=0 and
# attention_mask=1. The first version of this script assumed key/value were interleaved per layer,
# which scrambles every layer above 0 and made the seeded step look like a compiler failure.
N_LAYER = len(layers)
if 2 + 2 * N_LAYER != 38:
    raise SystemExit('expected 18 layers so that 2 + 2*18 = 38 declared inputs, got %d' % N_LAYER)
for li, (k, v) in enumerate(layers):
    kh = np.repeat(k.float().numpy(), args.rep, axis=1)
    vh = np.repeat(v.float().numpy(), args.rep, axis=1)
    for letter, arr, base in (('key', kh, 2), ('value', vh, 2 + N_LAYER)):
        buf = np.zeros((1, arr.shape[1], MAXSLOT, arr.shape[-1]), dtype=np.float16)
        for t in range(arr.shape[2]):
            buf[0, :, t + args.slot_base, :] = arr[0, :, t, :].astype(np.float16)
        v_id = base + li
        buf.tofile(os.path.join(args.out, 'in_v%d.ftmp' % v_id))
        n_written += 1
print('MAPPING layer %d key -> v%d .. layer %d key -> v%d | layer 0 value -> v%d .. layer %d value -> v%d'
      % (0, 2, N_LAYER - 1, 2 + N_LAYER - 1, 2 + N_LAYER, N_LAYER - 1, 2 + 2 * N_LAYER - 1), flush=True)

json.dump({'prompt': PROMPT, 'n_cached': args.n, 'next_token': PROMPT[args.n],
           'slot_base': args.slot_base, 'past_len_feed': args.n + args.slot_base,
           'gold_argmax': int(np.argmax(gold)),
           'gold_top5': [int(t) for t in np.argsort(gold)[-5:][::-1]],
           'files': n_written},
          open(os.path.join(args.out, 'meta.json'), 'w', encoding='utf-8'), indent=1)
np.save(os.path.join(args.out, 'gold_next.npy'), gold)
print('WROTE %d cache files to %s (slot_base=%d, cache holds %d tokens, feed past_len=%d)'
      % (n_written, args.out, args.slot_base, args.n, args.n + args.slot_base), flush=True)
print('GOLD_NEXT argmax %d top5 %s' % (int(np.argmax(gold)),
      [int(t) for t in np.argsort(gold)[-5:][::-1]]), flush=True)
print('DONE_SEED')
