"""Make a seed variant whose pad slot is *not* zero, to test whether the graph actually attends to it.

host_gen.py has to start the sequence at slot 1 with slot 0 left as an all-zero pad -- feeding element 1 = 0
makes the dynamic slice blow up (rc=1 here too, same op as the Qwen3 export). If nothing masks slot 0, its
zero key gets a score of q.0 = 0 and its zero value joins the softmax denominator, which dilutes every
attention output by a factor that is worst at step 0 (1 real token + 1 pad = half weight on nothing) and
decays as the sequence grows. That is the shape of the error the chained run shows: cos -0.255 at step 0,
0.96 by step 6.

Discriminator: write nonzero values into slot 0 of every past tensor. If the logits move, the pad is being
attended and the chained drift is that dilution; if they are identical to the clean seed, something masks
it and the drift has another cause.

Usage: python -X utf8 mk_padrand.py --src seed6_a --dst seed6_pad --slot 0
(seed6_a was built with slot_base=1, so slot 0 is the pad in every one of its 36 tensors.)
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--slot', type=int, default=0, help='which slot to randomise')
ap.add_argument('--seed', type=int, default=20261030)
args = ap.parse_args()

os.makedirs(args.dst, exist_ok=True)
rng = np.random.default_rng(args.seed)
files = [f for f in os.listdir(args.src) if f.startswith('in_v') and f.endswith('.ftmp')]
if not files:
    raise SystemExit('no in_v*.ftmp in %s' % args.src)
moved = 0
was_zero = 0
for f in files:
    a = np.fromfile(os.path.join(args.src, f), dtype='<f2').reshape(1, 8, 512, 128)
    before = a[0, :, args.slot].copy()
    a[0, :, args.slot, :] = rng.normal(0, 1, size=(8, 128)).astype(np.float16)
    was_zero += int(np.abs(before).max() == 0.0)
    a.tofile(os.path.join(args.dst, f))
for f in ('gold_next.npy', 'meta.json'):
    s = os.path.join(args.src, f)
    if os.path.exists(s):
        open(os.path.join(args.dst, f), 'wb').write(open(s, 'rb').read())
print("%d tensors rewritten at slot %d; %d of them had an all-zero slot beforehand"
      % (len(files), args.slot, was_zero))
print('DONE_PADRAND')
