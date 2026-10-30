"""Score the pad-repaired walk of the *compiled* E41 net against HF float32, on the school machine.

`--mode check` in eval_ernie.py proved the ORT twin reproduces the compiled net step for step, so the twin
can predict what the slot1->slot0 repair should do; this is the other half. It reads the logits the iCraft
Host backend actually produced for the repaired walk (pushed from the compile host as
compiled_padfix/step<i>/logits.npy) and scores them next to the unrepaired walk already on disk
(compiled_chain/step<i>/logits.npy), with the twin's own plain/repaired finals as the expectation.

Three controls run before the numbers:
  step 0 equality  -- the repaired walk's step 0 must be byte-identical to the plain walk's, because both
                      start from an all-zero cache and the same token; if it is not, the driver changed
                      something other than the repair.
  argmax chain     -- each walk's per-step argmax is recomputed from its own logits.npy (not read from the
                      driver's steps.json) and compared with the twin's, so a convention difference between
                      the two implementations shows up here rather than in the error columns.
  convention       -- the compiled driver mirrors slot 1 into slot 0 on *every* step, the twin below mirrors
                      it twice (the eval_ernie.py padfix convention). If the argmax control passes, the two
                      agree on this prompt; the difference is not silently averaged away.

Usage (school):
  python eval_padfix_compiled.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import onnxruntime_genai  # noqa: F401  registers the com.microsoft custom ops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = r'D:\yzd\WCIS\MOD\ernie'
SRC = os.path.join(ROOT, 'ERNIE-4.5-0.3B-PT')
E41_ORT = os.path.join(ROOT, 'e_v41', 'model_ort.onnx')
PLAIN = os.path.join(ROOT, 'eval', 'compiled_chain')
FIXED = os.path.join(ROOT, 'eval', 'compiled_padfix')
PROMPT = '请用一句话介绍量子计算。'
HEADS, SLOTS, HD, N_LAYER = 8, 512, 128, 18

torch.set_num_threads(os.cpu_count() or 12)
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(E41_ORT, so, providers=['CPUExecutionProvider'])
INS = [i.name for i in sess.get_inputs()]
OUTS = [o.name for o in sess.get_outputs()]
tok = AutoTokenizer.from_pretrained(SRC)
hf32 = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).eval()

def step_dirs(d):
    return sorted([int(f[4:]) for f in os.listdir(d)
                   if f.startswith('step') and os.path.isdir(os.path.join(d, f))])


ids = tok(PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
n_plain = len(step_dirs(PLAIN))
n_fixed = len(step_dirs(FIXED))
print('prompt %d tokens %s' % (len(ids), ids), flush=True)
print('compiled steps: plain %d (%s) padfix %d (%s)' % (n_plain, PLAIN, n_fixed, FIXED), flush=True)


def step_logits(d, i):
    return np.load(os.path.join(d, 'step%d' % i, 'logits.npy')).astype(np.float64).ravel()


def score(a, b):
    return {'cos': round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 6),
            'maxabs': round(float(np.abs(a - b).max()), 4),
            'tv': round(float(0.5 * np.abs(np.exp(a - a.max()) / np.exp(a - a.max()).sum()
                                           - np.exp(b - b.max()) / np.exp(b - b.max()).sum()).sum()), 6),
            'top1_match': bool(int(a.argmax()) == int(b.argmax())),
            'rank_of_ref_top1': int((a > a[int(b.argmax())]).sum()),
            'top1': int(a.argmax())}


def zero_cache():
    z = np.zeros((1, HEADS, SLOTS, HD), dtype=np.float16)
    return {n: z.copy() for n in INS if n.startswith('past_key_values.')}


def twin_step(cache, token_id, past_len):
    feed = {'input_ids': np.array([[token_id]], dtype=np.int64),
            'attention_mask': np.array([[1, past_len, 0]], dtype=np.int64)}
    feed.update(cache)
    res = sess.run(None, feed)
    lg = np.asarray(res[OUTS.index('logits')]).astype(np.float64).ravel()
    out = {}
    for i, name in enumerate(OUTS):
        if name == 'logits':
            continue
        tgt = 'past_key_values.' + name[len('present.'):]
        if tgt in cache:
            out[tgt] = np.asarray(res[i]).astype(np.float16).copy()
    return lg, out


def twin_walk(repair):
    cache = zero_cache()
    lgs, argmaxes = [], []

    def seed_pad():
        for nm in cache:
            cache[nm][0, :, 0, :] = cache[nm][0, :, 1, :]

    seq = ([ids[0]] + ids) if repair else list(ids)
    for i, t in enumerate(seq):
        past_len = (1 if i <= 1 else i) if repair else (i + 1)
        lg, upd = twin_step(cache, t, past_len)
        lgs.append(lg)
        argmaxes.append(int(lg.argmax()))
        cache.update(upd)
        if repair and i <= 1:
            seed_pad()
    return lgs, argmaxes


with torch.no_grad():
    gold = hf32(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float().numpy().astype(np.float64)

print('CONTROL step0 identical plain vs padfix: %s'
      % bool(np.array_equal(step_logits(PLAIN, 0), step_logits(FIXED, 0))), flush=True)
c_plain_all = [int(step_logits(PLAIN, i).argmax()) for i in range(n_plain)]
c_fixed_all = [int(step_logits(FIXED, i).argmax()) for i in range(n_fixed)]
print('compiled plain  per-step argmax %s' % c_plain_all, flush=True)
print('compiled padfix per-step argmax %s' % c_fixed_all, flush=True)

t_plain, t_plain_arg = twin_walk(False)
t_fixed, t_fixed_arg = twin_walk(True)
c_plain = step_logits(PLAIN, n_plain - 1)
c_fixed = step_logits(FIXED, n_fixed - 1)
print('CONTROL compiled plain  argmax chain == twin plain: %s  (%s vs %s)'
      % (c_plain_all == t_plain_arg, c_plain_all, t_plain_arg), flush=True)
print('CONTROL compiled padfix argmax chain == twin padfix: %s  (%s vs %s)'
      % (c_fixed_all == t_fixed_arg, c_fixed_all, t_fixed_arg), flush=True)

rows = [('compiled plain', c_plain), ('compiled padfix', c_fixed),
        ('twin plain', t_plain[-1]), ('twin padfix', t_fixed[-1]),
        ('padfix vs plain', None)]
print('%-16s %9s %8s %8s %6s %6s %7s' % ('column', 'cos', 'maxabs', 'TV', 'top1==hf', 'rank', 'top1'),
      flush=True)
for name, lg in rows[:4]:
    s = score(lg, gold)
    print('%-16s %9.6f %8.4f %8.5f %6s %6d %7d'
          % (name, s['cos'], s['maxabs'], s['tv'], s['top1_match'], s['rank_of_ref_top1'], s['top1']),
          flush=True)
d = score(c_fixed, c_plain)
print('%-16s %9.6f %8.4f %8.5f %6s %6d %7d  (fixed measured against plain, not against HF)'
      % ('padfix vs plain', d['cos'], d['maxabs'], d['tv'], d['top1_match'], d['rank_of_ref_top1'],
         d['top1']), flush=True)
print('twin predicted shift in final argmax: %d -> %d   compiled: %d -> %d'
      % (t_plain_arg[-1], t_fixed_arg[-1], int(c_plain.argmax()), int(c_fixed.argmax())), flush=True)
print('HF top5 %s' % [int(t) for t in np.argsort(gold)[-5:][::-1]], flush=True)
print('decoded: plain  %r' % tok.decode(ids + [int(c_plain.argmax())]), flush=True)
print('decoded: padfix %r' % tok.decode(ids + [int(c_fixed.argmax())]), flush=True)
print('DONE_PADFIX_COMPILED', flush=True)
