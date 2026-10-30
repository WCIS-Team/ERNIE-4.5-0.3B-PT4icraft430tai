"""Reorder the K axis of an HF-seeded cache feed to match a e_v41-style graph.

permute_rope_heads.py moves Q/K head-dim d to new position, so the post-rotary K that GQA writes into
present.key is in that order too. An HF-derived K cache has to be moved the same way or the seeded
step compares two different dim orders. V is never rotated, so it and the padded dead slots stay put.

Cache inputs are declared keys-first, values-last: v2..v19 are past_key_values.0..17.key, v20..v37 the
values, v38..v41 the declared-but-unread pads.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--rows', type=int, default=512)
    ap.add_argument('--head-dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=18)
    args = ap.parse_args()

    hd = args.head_dim
    perm = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
    os.makedirs(args.dst, exist_ok=True)

    moved = kept = 0
    for f in sorted(os.listdir(args.src)):
        s = os.path.join(args.src, f)
        d = os.path.join(args.dst, f)
        if not f.startswith('in_v') or not f.endswith('.ftmp'):
            shutil.copyfile(s, d)
            continue
        v = int(f[4:-5])
        nbytes = args.heads * args.rows * hd * 2
        if os.path.getsize(s) != nbytes:
            raise SystemExit('%s is %d bytes, expected %d' % (s, os.path.getsize(s), nbytes))
        if 2 <= v < 2 + args.layers:
            a = np.fromfile(s, dtype=np.float16).reshape(1, args.heads, args.rows, hd)
            a[..., perm].astype(np.float16).tofile(d)
            moved += 1
        else:
            shutil.copyfile(s, d)
            kept += 1
    shutil.copyfile(os.path.join(args.src, 'meta.json'), os.path.join(args.dst, 'meta.json'))
    shutil.copyfile(os.path.join(args.src, 'gold_next.npy'), os.path.join(args.dst, 'gold_next.npy'))
    print('K slots permuted %d (v2..v%d); files copied unchanged %d (V and pads)'
          % (moved, 1 + args.layers, kept))
    a = np.fromfile(os.path.join(args.dst, 'in_v2.ftmp'), dtype=np.float16).reshape(-1, hd)
    b = np.fromfile(os.path.join(args.src, 'in_v2.ftmp'), dtype=np.float16).reshape(-1, hd)
    print('check: new[0, %d] == old[0, 1] -> %s; V unchanged -> %s'
          % (hd // 2, bool(a[0, hd // 2] == b[0, 1]),
             np.array_equal(np.fromfile(os.path.join(args.dst, 'in_v20.ftmp'), dtype=np.float16),
                            np.fromfile(os.path.join(args.src, 'in_v20.ftmp'), dtype=np.float16))))
    print('DONE_PERMSEED')


if __name__ == '__main__':
    main()
