"""Print the io contract and step latency of the two ORT graphs the school evaluation will run.

The eval script has to feed the ported graph by *declared index* (the compiled net is fed that way, and
the four dead pad inputs sit at the end of the list), so the name list has to be read before that mapping
is written down rather than assumed. Latency is measured here too: it decides how many items the suite can
carry, because the board-condition run needs one graph evaluation per prompt token.

Usage (school): python probe_e41_ort.py
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_genai  # noqa: F401  registers the com.microsoft custom ops

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

E41 = r'D:\yzd\WCIS\MOD\ernie\e_v41\model_ort.onnx'
NATIVE = r'D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda\model.onnx'
CFG = r'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT\config.json'
HEADS, SLOTS, HD = 8, 512, 128

cfg = json.load(open(CFG, encoding='utf-8'))
print('CFG layers=%d heads=%d kv=%d head_dim=%s hidden=%d vocab=%d'
      % (cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads'],
         cfg.get('head_dim'), cfg['hidden_size'], cfg['vocab_size']), flush=True)

so = ort.SessionOptions()
so.log_severity_level = 3
so.intra_op_num_threads = 12

sess = ort.InferenceSession(E41, so, providers=['CPUExecutionProvider'])
ins = [i.name for i in sess.get_inputs()]
outs = [o.name for o in sess.get_outputs()]
print('E41 INPUTS %d' % len(ins), flush=True)
for n, i in enumerate(sess.get_inputs()):
    print('  v%-2d %-32s %-18s %s' % (n, i.name, i.type, i.shape), flush=True)
print('E41 OUTPUTS %d %s' % (len(outs), outs[:6]), flush=True)

feed = {'input_ids': np.array([[94515]], dtype=np.int64),
        'attention_mask': np.array([[1, 1, 0]], dtype=np.int64)}
for n in ins:
    if n.startswith('past_key_values.'):
        feed[n] = np.zeros((1, HEADS, SLOTS, HD), dtype=np.float16)
t0 = time.time()
res = sess.run(None, feed)
print('E41 STEP0 %.1f s  outputs %d  logits %s argmax %d top5 %s'
      % (time.time() - t0, len(res), np.asarray(res[0]).shape,
         int(np.asarray(res[0]).ravel().argmax()),
         [int(t) for t in np.argsort(np.asarray(res[0]).astype(np.float64).ravel())[-5:][::-1]]),
      flush=True)

t0 = time.time()
res = sess.run(None, feed)
print('E41 STEP0 repeat %.1f s' % (time.time() - t0), flush=True)

ns = ort.InferenceSession(NATIVE, so, providers=['CPUExecutionProvider'])
print('NATIVE INPUTS %d %s' % (len(ns.get_inputs()), [i.name for i in ns.get_inputs()][:4]), flush=True)
print('NATIVE OUTPUTS %d %s' % (len(ns.get_outputs()), [o.name for o in ns.get_outputs()][:4]), flush=True)
for i in ns.get_inputs():
    if i.name in ('input_ids', 'attention_mask', 'past_key_values.0.key', 'position_ids'):
        print('  %-26s %-18s %s' % (i.name, i.type, i.shape), flush=True)
print('NATIVE_KV_TYPE', [i.type for i in ns.get_inputs() if i.name == 'past_key_values.0.key'], flush=True)
print('DONE_PROBE', flush=True)
