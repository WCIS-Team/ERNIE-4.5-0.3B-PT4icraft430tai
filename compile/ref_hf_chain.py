"""HF golden chain for the ERNIE prompt, position by position, so the iCraft Host logits have a reference.

Teacher-forced then greedy: steps 0..len(prompt)-1 feed the fixed prompt prefix, so those logits are
directly comparable with the compiled net's step-by-step argmax. From there the script extends with its
own argmax, and the caller can point host_gen.py at the same ids (--prompt "prompt...,gen...") to keep the
two chains on identical inputs instead of letting them drift apart.

use_cache=False over the whole prefix, exactly like verify_onnx_ernie.py's golden() -- the check is
whether the compiled graph reproduces the reference, so the reference must not lean on the same KV-cache
mechanism it is being tested against.

Run on the compile-of-record host (school): python ref_hf_chain.py --prompt-file ... --out ...

Outputs: <out>/step<i>_gold.npy (last-position logits, float32), <out>/chain.json (per-step argmax,
top-5, and the decoded text so far).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=r'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT')
ap.add_argument('--prompt-ids', default='', help='comma separated; empty means --prompt text')
ap.add_argument('--prompt', default='请用一句话介绍量子计算。')
ap.add_argument('--max-new', type=int, default=5)
ap.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
ap.add_argument('--out', required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.src)
if args.prompt_ids:
    prompt = [int(t) for t in args.prompt_ids.split(',') if t.strip()]
else:
    prompt = tok(args.prompt, return_tensors='pt').input_ids[0].tolist()
print('PROMPT', len(prompt), prompt, flush=True)
print('PROMPT_TEXT', repr(tok.decode(prompt)), flush=True)

dt = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
model = AutoModelForCausalLM.from_pretrained(args.src, dtype=dt).eval()
print('DT', args.dtype, flush=True)

seq = list(prompt)
record = []
for i in range(len(prompt) + args.max_new):
    t0 = time.time()
    ids = torch.tensor([seq[:i + 1]])
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    lg = out.logits[0, -1].float().numpy()
    np.save(os.path.join(args.out, 'step%d_gold.npy' % i), lg)
    top5 = [int(t) for t in np.argsort(lg)[-5:][::-1]]
    nxt = top5[0]
    rec = {'step': i, 'fed': seq[i], 'n_prompt_left': i < len(prompt) - 1,
           'argmax': nxt, 'top5': top5,
           'logits_min': float(lg.min()), 'logits_max': float(lg.max()),
           'text': tok.decode(seq[:i + 1]), 'token': tok.decode([seq[i]]),
           'seconds': round(time.time() - t0, 1)}
    record.append(rec)
    # extend only once the prompt is exhausted: appending during the prompt phase would drop the
    # generation argmaxes into seq[len(prompt)..] and the later slices would feed those instead
    if i >= len(prompt) - 1:
        seq.append(nxt)
    print('step %2d fed %-7d -> %-7d  tok=%-8r  top5=%s  (%.1fs)'
          % (i, seq[i], nxt, tok.decode([seq[i]]), top5, time.time() - t0), flush=True)
    json.dump(record, open(os.path.join(args.out, 'chain.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)

print('GREEDY_TEXT', tok.decode(seq), flush=True)
print('DONE_HF_CHAIN', flush=True)
