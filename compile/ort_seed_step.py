"""Control run: the *rewritten* ONNX through onnxruntime-genai, on the same HF-seeded cache the compiled
net was fed, so the two errors can be told apart.

The compiled decoder, given HF's own cache for the first 6 prompt tokens and the 7th token as input,
reproduces the golden top-1 (23) and the golden's top-4 set, but only to cos 0.965 / maxabs 4.2 -- about
2.7x the maxabs of the pre-rewrite ONNX check that was recorded in the export log (maxabs 1.57, cos 0.991).
Two different things could be responsible: the ERNIE graph rewrites this pipeline performed
(split_qkv -> tile_kv kv2->8 -> rotary_interleaved=0 -> shrink_rope), or icraft's five compile stages.
Running the rewritten .onnx itself, on the identical feed, splits that: same error as the compiled net
means the rewrites own it, smaller error means the compiler does.

Usage (school):
  python ort_seed_step.py --onnx D:\yzd\WCIS\MOD\ernie\e_v35 --seed D:\yzd\WCIS\MOD\ernie\seed6_a \
      --token 93977 --tdf 1,7,0 --cmp "S:\...\gen_e38_seed_a\step0\logits.npy"
--cmp takes the compiled net's logits file; it is compared against both this run and the golden.
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import onnxruntime_genai  # noqa: F401  registers the com.microsoft custom ops

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ap = argparse.ArgumentParser()
ap.add_argument('--onnx', required=True, help='directory holding model.onnx + model.onnx.data')
ap.add_argument('--model', default='model.onnx', help='graph file inside --onnx '
                                                      '(model_ort.onnx = the ORT-loadable twin)')
ap.add_argument('--seed', required=True, help='directory of in_v<id>.ftmp cache feeds, v_id = declared index')
ap.add_argument('--token', type=int, required=True)
ap.add_argument('--tdf', default='1,7,0', help='the [1,3] attention_mask triple, same meaning as host_gen')
ap.add_argument('--gold', default='', help='golden logits .npy; defaults to <seed>/gold_next.npy')
ap.add_argument('--cmp', default='')
ap.add_argument('--heads', type=int, default=8,
                help='axis-1 size of this graph cache tensors: 8 after tile_kv_heads replication, 2 native')
args = ap.parse_args()

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(os.path.join(args.onnx, args.model), so,
                            providers=['CPUExecutionProvider'])
ins = sess.get_inputs()
outs = sess.get_outputs()
print('INPUTS %d OUTPUTS %s' % (len(ins), [o.name for o in outs]), flush=True)

by_name = {i.name: i for i in ins}
feed = {}
for i in ins:
    n = i.name
    if n == 'input_ids':
        feed[n] = np.array([[args.token]], dtype=np.int64)
    elif n == 'attention_mask':
        feed[n] = np.array([[int(x) for x in args.tdf.split(',')]], dtype=np.int64)
    elif n.startswith('past_key_values.'):
        idx = [k for k, x in enumerate(ins) if x.name == n][0]
        p = os.path.join(args.seed, 'in_v%d.ftmp' % idx)
        if not os.path.exists(p):
            print('  %s (v%d) not seeded -> zeros' % (n, idx), flush=True)
            shape = [1 if x == 'batch_size' else (512 if x == 'past_sequence_length' else x)
                     for x in i.shape]
            feed[n] = np.zeros(shape, dtype=np.float16)
        else:
            arr = np.fromfile(p, dtype='<f2')
            feed[n] = arr.reshape(1, args.heads, 512, 128)
    else:
        raise SystemExit('unexpected graph input %r' % n)

lg = sess.run(['logits'], feed)[0]
h = np.asarray(lg).astype(np.float64).ravel()
print('RUN ok shape %s argmax %d top5 %s' % (np.asarray(lg).shape, int(np.argmax(h)),
      [int(t) for t in np.argsort(h)[-5:][::-1]]), flush=True)


def score(name, a, b):
    d = a - b
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print('%-22s maxabs %8.3f relmax %.4f cos %.5f  argmax %6d vs %6d  rank_of_b_top1_in_a %d'
          % (name, np.abs(d).max(), np.abs(d).max() / np.abs(b).max(), cos,
             int(np.argmax(a)), int(np.argmax(b)), int((a > a[int(np.argmax(b))]).sum())), flush=True)


gold = np.load(args.gold or os.path.join(args.seed, 'gold_next.npy')).astype(np.float64).ravel()
score('ORT-rewritten vs HF', h, gold)
if args.cmp:
    c = np.load(args.cmp).astype(np.float64).ravel()
    score('ORT-rewritten vs compiled', h, c)
    score('compiled vs HF', c, gold)
print('DONE_ORT_SEED', flush=True)
