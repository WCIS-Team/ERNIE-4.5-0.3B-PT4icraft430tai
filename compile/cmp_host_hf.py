"""Compare the iCraft Host-backend logits against the HF golden, step by step, teacher-forced.

Both sides are indexed by step number: host step i feeds prompt[i] with the KV chain built by steps
0..i-1, HF golden step i is the last-position logits of the same i+1 token prefix computed with
use_cache=False. So a mismatch is arithmetic or graph-level, not a difference in decoding policy.

Reported per step: max abs difference, that difference relative to the golden's own spread (max|gold|
is ~20 here, so a raw maxabs of 1.5 means something only next to the logit scale), cosine of the two
vectors, and the *rank* each side's argmax has on the other -- rank 0 means both agree on the winner,
rank <=4 means the reference answer is still inside the top-5 and the gap is noise, rank in the
thousands means a structural break (wrong head mapping, wrong position, wrong cache layout).

Usage: python -X utf8 cmp_host_hf.py --host <gen_e38 dir> --gold <hf_chain_e38 dir> --steps 7
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ap = argparse.ArgumentParser()
ap.add_argument('--host', required=True)
ap.add_argument('--gold', required=True)
ap.add_argument('--steps', type=int, default=7)
ap.add_argument('--host-file', default='logits.npy')
ap.add_argument('--gold-file', default='step%d_gold.npy')
args = ap.parse_args()


def rank_of(vec, tok):
    return int((vec > vec[tok]).sum())


rows = []
for i in range(args.steps):
    h = np.load(os.path.join(args.host, 'step%d' % i, args.host_file)).astype(np.float64).ravel()
    g = np.load(os.path.join(args.gold, args.gold_file % i)).astype(np.float64).ravel()
    if h.shape != g.shape:
        raise SystemExit('step %d: host %s vs gold %s' % (i, h.shape, g.shape))
    hd, gd = h - g, float(np.abs(g).max())
    cos = float(h @ g / (np.linalg.norm(h) * np.linalg.norm(g)))
    th = np.exp(h - h.max()); th /= th.sum()
    tg = np.exp(g - g.max()); tg /= tg.sum()
    kt = int(np.argmax(g))
    rows.append({
        'step': i, 'argmax_h': int(np.argmax(h)), 'argmax_g': kt,
        'maxabs': float(np.abs(hd).max()), 'relmax': float(np.abs(hd).max() / gd),
        'meanabs': float(np.abs(hd).mean()), 'cos': cos,
        'std_h': float(h.std()), 'std_g': float(g.std()),
        'rank_of_gold_top1_in_host': rank_of(h, kt),
        'rank_of_host_top1_in_gold': rank_of(g, int(np.argmax(h))),
        'top5_h': [int(t) for t in np.argsort(h)[-5:][::-1]],
        'top5_g': [int(t) for t in np.argsort(g)[-5:][::-1]],
        'p_gold_top1_under_host': float(th[kt]),
        'kl_g_h': float((tg * (np.log(tg) - np.log(th))).sum()),
    })

print('step  host  gold  maxabs   relmax      cos   std_h  std_g  rankG@H rankH@G  p_g@H     KL')
for r in rows:
    print('%3d %6d %6d %8.3f %8.4f %8.5f %6.2f %6.2f %8d %8d %7.3f %7.2f'
          % (r['step'], r['argmax_h'], r['argmax_g'], r['maxabs'], r['relmax'], r['cos'],
             r['std_h'], r['std_g'], r['rank_of_gold_top1_in_host'],
             r['rank_of_host_top1_in_gold'], r['p_gold_top1_under_host'], r['kl_g_h']))
print()
for r in rows:
    print('step %d top5 host %s | gold %s' % (r['step'], r['top5_h'], r['top5_g']))
print('MEAN relmax %.4f | MEAN cos %.5f | MEAN KL %.2f | top1 agree %d/%d'
      % (np.mean([r['relmax'] for r in rows]), np.mean([r['cos'] for r in rows]),
         np.mean([r['kl_g_h'] for r in rows]),
         sum(r['argmax_h'] == r['argmax_g'] for r in rows), len(rows)))
