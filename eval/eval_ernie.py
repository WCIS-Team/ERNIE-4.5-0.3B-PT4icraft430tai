"""School-side evaluation of the ported ERNIE graph: numeric fidelity, task accuracy, usability.

Why each column exists. The ported graph differs from the exporter's own ONNX in four ways at once --
fp16 weights, the Split/Concat that tiles the 2 KV heads to 8, the shrunken rope cache, and the head_dim
axis permutation that e_v41 carries to work around iCraft ignoring rotary_interleaved -- and the board
condition adds a fifth (prompt tokens walked one at a time over a 512-slot cache whose unused half is
all-zero and unmasked). A single error number therefore says nothing about which of those costs what. So
the suite scores the same items five ways:

  hf32         Hugging Face, float32, one full forward, use_cache=False -- the reference.
  hf16         same model, float16 -- what plain fp16 arithmetic costs, with no ONNX involved.
  native       onnxruntime on the exporter's own model.onnx -- what the official export path costs.
  e41_seeded   onnxruntime on the ORT twin of e_v41, one decode step on HF's own cache for prompt[:-1]
               -- isolates the four graph rewrites from the chained walk.
  e41_chain    onnxruntime on the same twin, cache built one token at a time from all zeros -- the exact
               feed the board driver will produce, so this is the column that answers 能不能用.

`--mode check` first proves the substitution: it re-runs the 7-token reference prompt as e41_chain and
compares against the compiled ZG net's own logits (measured on the iCraft Host backend, pushed here as
compiled_chain/step<i>/logits.npy). If the twin reproduces those, everything below scored on the twin
applies to the compiled net, and the twin is used because it runs in milliseconds rather than 11.5 s.

Cache feed convention (read back out of the compiled chain's own in_v*.ftmp, not assumed): input v_id 0 =
input_ids int64, v_id 1 = attention_mask as the 3-value token_dynamic_fm [1, past_len, 0] with past_len
counting the token being fed, v_id 2..19 = past_key_values.0..17.key, 20..37 = the values, 38..41 = the
four declared-but-unread pads, which still have to be fed. A prompt token at index t is written to slot
t+1 and slot 0 stays zero; the graph's own present.* buffers are re-fed unchanged, which keeps that offset.

Usage (school):
  python eval_ernie.py --mode check
  python eval_ernie.py --mode items --items items.json --out eval_items.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_genai  # noqa: F401  registers the com.microsoft custom ops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = r'D:\yzd\WCIS\MOD\ernie'
SRC = os.path.join(ROOT, 'ERNIE-4.5-0.3B-PT')
E41_ORT = os.path.join(ROOT, 'e_v41', 'model_ort.onnx')
NATIVE = os.path.join(ROOT, 'ernie_fp16_cuda', 'model.onnx')
CHAIN = os.path.join(ROOT, 'eval', 'compiled_chain')
PROMPT = '请用一句话介绍量子计算。'
REF_IDS = [94515, 94036, 31282, 5502, 44284, 3458, 93977]
HEADS, SLOTS, HD, N_LAYER = 8, 512, 128, 18

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True,
                choices=['check', 'items', 'sweep', 'pad', 'decomp', 'cachemap', 'padfix'])
ap.add_argument('--items', default=os.path.join(ROOT, 'eval', 'items.json'))
ap.add_argument('--out', default='')
ap.add_argument('--limit', type=int, default=0, help='score only the first N items (smoke run)')
ap.add_argument('--chat', action='store_true',
                help='wrap every prompt in the model\'s own chat template before tokenising')
args = ap.parse_args()

torch.set_num_threads(os.cpu_count() or 12)
PERM = np.concatenate([np.arange(0, HD, 2), np.arange(1, HD, 2)])

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(E41_ORT, so, providers=['CPUExecutionProvider'])
E41_INS = [i.name for i in sess.get_inputs()]
E41_OUTS = [o.name for o in sess.get_outputs()]
assert len(E41_INS) == 42, len(E41_INS)
# the 40 cache inputs are named past_key_values.L.X and their partners present.L.X; the four pads at
# L=18,19 have no output and stay zero, but still have to be in the feed

nsess = None
if args.mode in ('items', 'sweep'):
    nsess = ort.InferenceSession(NATIVE, so, providers=['CPUExecutionProvider'])
    NAT_OUTS = [o.name for o in nsess.get_outputs()]

tok = AutoTokenizer.from_pretrained(SRC)
hf32 = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).eval()
hf16 = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16).eval()
print('MODEL vocab=%d threads=%d' % (hf32.config.vocab_size, torch.get_num_threads()), flush=True)


def as_prompt(text):
    """Card says Training Stage = Posttraining, so the bare-cloze condition is out of distribution; --chat
    re-scores the same items inside the template the model was finished with."""
    if not args.chat:
        return text
    return tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                    add_generation_prompt=True)


def zero_cache():
    z = np.zeros((1, HEADS, SLOTS, HD), dtype=np.float16)
    return {n: z.copy() for n in E41_INS if n.startswith('past_key_values.')}


def e41_step(cache, token_id, past_len):
    feed = {'input_ids': np.array([[token_id]], dtype=np.int64),
            'attention_mask': np.array([[1, past_len, 0]], dtype=np.int64)}
    feed.update(cache)
    res = sess.run(None, feed)
    lg = np.asarray(res[E41_OUTS.index('logits')]).astype(np.float64).ravel()
    out = {}
    for i, name in enumerate(E41_OUTS):
        if name == 'logits':
            continue
        tgt = 'past_key_values.' + name[len('present.'):]
        if tgt in cache:
            out[tgt] = np.asarray(res[i]).astype(np.float16).copy()
    return lg, out


def e41_chain(ids):
    """Walk the prompt one token at a time over an all-zero cache -- the board driver's condition."""
    cache = zero_cache()
    logits = []
    for i, t in enumerate(ids):
        lg, upd = e41_step(cache, t, i + 1)
        logits.append(lg)
        cache.update(upd)
    return logits


def seed_cache(ids, gold_model):
    """HF's own cache for ids[:-1], tiled 2->8 and put in e_v41's permuted head_dim basis, slot t+1."""
    with torch.no_grad():
        out = gold_model(input_ids=torch.tensor([ids[:-1]]), use_cache=True)
    pkv = out.past_key_values
    if hasattr(pkv, 'layers'):
        layers = [(lyr.keys, lyr.values) for lyr in pkv.layers]
    else:
        layers = list(zip(pkv[0], pkv[1]))
    assert len(layers) == N_LAYER, len(layers)
    cache = zero_cache()
    for li, (k, v) in enumerate(layers):
        kh = np.repeat(k.float().numpy(), 4, axis=1).astype(np.float16)
        vh = np.repeat(v.float().numpy(), 4, axis=1).astype(np.float16)
        kh = kh[..., PERM]
        for letter, arr, base in (('key', kh, 0), ('value', vh, N_LAYER)):
            buf = cache['past_key_values.%d.%s' % (li, letter)]
            for t in range(arr.shape[2]):
                buf[0, :, t + 1, :] = arr[0, :, t, :]
    return cache


def e41_seeded(ids):
    cache = seed_cache(ids, hf32)
    lg, _ = e41_step(cache, ids[-1], len(ids))
    return lg


def native_rows(ids):
    n_kv = hf32.config.num_key_value_heads
    feed = {'input_ids': np.array([ids], dtype=np.int64),
            'attention_mask': np.ones((1, len(ids)), dtype=np.int64)}
    for i in range(N_LAYER):
        for letter in ('key', 'value'):
            feed['past_key_values.%d.%s' % (i, letter)] = np.zeros((1, n_kv, 0, HD), dtype=np.float16)
    res = nsess.run(None, feed)
    return np.asarray(res[NAT_OUTS.index('logits')]).astype(np.float64)[0]


def native_full(ids):
    return native_rows(ids)[-1]


def hf_full(model, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    return out.logits[0].float().numpy().astype(np.float64)


def score(a, b):
    """a scored against reference b: absolute, angular and information-theoretic agreement."""
    d = float(np.abs(a - b).max())
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    pa = np.exp(a - a.max()); pa /= pa.sum()
    pb = np.exp(b - b.max()); pb /= pb.sum()
    kl = float((pb * np.log((pb + 1e-12) / (pa + 1e-12))).sum())
    return {'maxabs': round(d, 4), 'relmax': round(d / float(np.abs(b).max() + 1e-9), 5),
            'cos': round(cos, 6), 'kl': round(kl, 5),
            'top1_match': bool(int(a.argmax()) == int(b.argmax())),
            'rank_of_ref_top1': int((a > a[int(b.argmax())]).sum())}


def nll(lg, ids):
    """mean -log P over positions i>=1, lg[i] = logits after feeding ids[0..i-1] (list or full row set).

    The partition function is logsumexp over the whole vocabulary; without it the number is not a
    cross-entropy at all (the first version of this function reported the logit ceiling instead).
    """
    tot, cnt = 0.0, 0
    for i in range(1, len(ids)):
        row = np.asarray(lg[i - 1] if isinstance(lg, list) else lg[i - 1], dtype=np.float64)
        m = row.max()
        lse = m + np.log(np.exp(row - m).sum())
        tot += lse - row[ids[i]]
        cnt += 1
    return tot / max(cnt, 1), cnt


# ------------------------------------------------------------------ mode check
if args.mode == 'check':
    gold = hf_full(hf32, REF_IDS)                      # gold[i] = logits after ids[0..i-1]
    comp = [np.load(os.path.join(CHAIN, 'step%d' % i, 'logits.npy')).astype(np.float64).ravel()
            for i in range(len(REF_IDS))]
    t0 = time.time()
    ch = e41_chain(REF_IDS)
    dt = time.time() - t0
    print('ORT_CHAIN %d steps in %.3f s (%.1f ms/step)' % (len(REF_IDS), dt, 1000 * dt / len(REF_IDS)),
          flush=True)
    rows = []
    for i in range(len(REF_IDS)):
        r = {'step': i, 'fed': REF_IDS[i],
             'comp_vs_hf': score(comp[i], gold[i]),
             'ort_vs_hf': score(ch[i], gold[i]),
             'ort_vs_compiled': score(ch[i], comp[i]),
             'argmax_compiled': int(comp[i].argmax()), 'argmax_ort': int(ch[i].argmax()),
             'argmax_hf': int(gold[i].argmax())}
        rows.append(r)
        print('step %d fed %6d  ORT-vs-HF cos %s maxabs %s | ORT-vs-compiled cos %s maxabs %s | '
              'argmax C/O/HF %d/%d/%d'
              % (i, REF_IDS[i], r['ort_vs_hf']['cos'], r['ort_vs_hf']['maxabs'],
                 r['ort_vs_compiled']['cos'], r['ort_vs_compiled']['maxabs'],
                 r['argmax_compiled'], r['argmax_ort'], r['argmax_hf']), flush=True)
    json.dump(rows, open(os.path.join(ROOT, 'eval', 'check_e41.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print('DONE_CHECK', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ mode sweep
if args.mode == 'sweep':
    # Self-written plain prose. A pretrained base model is not a cloze engine (the 60-item suite shows its
    # own float32 top-1 misses the idiomatic completion), so the model's *ability* can't separate a good
    # port from a bad one. Continuation likelihood and greedy agreement can: they ask only whether the
    # ported net assigns the same probabilities to the same next tokens.
    ZH = ('今天早上下了小雨，街面上的水洼映着路灯。我撑着伞走过两个路口，卖早餐的摊子已经排起了队。'
          '摊主一边装豆浆，一边和熟客聊着这几天的天气。他说雨水来得早，地里的菜反而长得快。'
          '我听着觉得有道理，路上的人行色匆匆，谁也没有多停留。')
    EN = ('The morning was quiet, and a thin rain covered the street. Walking past the second crossing, '
          'I stopped at a small food stall where the owner was already serving the regulars. '
          'He said the early rain makes the vegetables grow faster, and I thought that sounded reasonable. '
          'Nobody lingered on the road, because everyone seemed to be in a hurry.')
    texts = {'zh': tok(ZH, return_tensors='pt', add_special_tokens=False).input_ids[0].tolist(),
             'en': tok(EN, return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()}
    out = {'texts': {k: len(v) for k, v in texts.items()}}

    print('SWEEP length scaling of the chained walk (final-step logits vs HF float32)', flush=True)
    rows = []
    for L in (4, 8, 16, 24, 32, 48, 64, len(texts['zh'])):
        sub = texts['zh'][:L]
        g = hf_full(hf32, sub)[-1]
        ls = e41_seeded(sub)
        lc = e41_chain(sub)[-1]
        r = {'L': L, 'seeded': score(ls, g), 'chain': score(lc, g),
             'argmax_hf': int(g.argmax()), 'argmax_seeded': int(ls.argmax()),
             'argmax_chain': int(lc.argmax())}
        rows.append(r)
        print('L=%3d  seeded cos %.5f maxabs %.3f rank_of_hf1 %3d | chain cos %.5f maxabs %.3f '
              'rank_of_hf1 %5d | argmax H/S/C %d/%d/%d'
              % (L, r['seeded']['cos'], r['seeded']['maxabs'], r['seeded']['rank_of_ref_top1'],
                 r['chain']['cos'], r['chain']['maxabs'], r['chain']['rank_of_ref_top1'],
                 r['argmax_hf'], r['argmax_seeded'], r['argmax_chain']), flush=True)
    out['length'] = rows

    print('SWEEP perplexity (mean -log P of the actual next token, whole text chained token by token)',
          flush=True)
    ppl = {}
    for k, ids in texts.items():
        g32 = hf_full(hf32, ids)
        n32, cnt = nll(g32, ids)
        n16, _ = nll(hf_full(hf16, ids), ids)
        nnat, _ = nll(native_rows(ids), ids)
        t0 = time.time()
        ch = e41_chain(ids)
        ms = 1000.0 * (time.time() - t0) / len(ids)
        nch, _ = nll(ch, ids)
        # per-position agreement of the chained walk with HF, over the whole text
        agree = sum(1 for i in range(1, len(ids))
                    if int(ch[i - 1].argmax()) == int(g32[i - 1].argmax()))
        ppl[k] = {'tokens': len(ids), 'scored': cnt, 'ppl_hf32': round(math.exp(n32), 3),
                  'ppl_hf16': round(math.exp(n16), 3), 'ppl_native': round(math.exp(nnat), 3),
                  'ppl_chain': round(math.exp(nch), 3),
                  'nll_hf32': round(n32, 4), 'nll_chain': round(nch, 4),
                  'chain_ms_per_step': round(ms, 1),
                  'argmax_agree_over_text': '%d/%d' % (agree, len(ids) - 1)}
        print('%s %d tokens  PPL hf32 %s | hf16 %s | native %s | chain %s  (chain %.1f ms/step, '
              'per-step argmax agreement %s)'
              % (k, len(ids), ppl[k]['ppl_hf32'], ppl[k]['ppl_hf16'], ppl[k]['ppl_native'],
                 ppl[k]['ppl_chain'], ppl[k]['chain_ms_per_step'],
                 ppl[k]['argmax_agree_over_text']), flush=True)
    out['ppl'] = ppl

    print('SWEEP free-running greedy continuation, chained net vs HF greedy', flush=True)
    gens = []
    for k, ids in texts.items():
        pre = ids[:12]
        seq_c, seq_h = list(pre), list(pre)
        cache = zero_cache()
        lg = None
        for i, t in enumerate(pre):
            lg, upd = e41_step(cache, t, i + 1)
            cache.update(upd)
        for _ in range(24):
            nxt = int(lg.argmax())
            seq_c.append(nxt)
            lg, upd = e41_step(cache, nxt, len(seq_c))
            cache.update(upd)
        for _ in range(24):
            gh = hf_full(hf32, seq_h)[-1]
            seq_h.append(int(gh.argmax()))
        m = min(len(seq_c), len(seq_h))
        agree = sum(1 for i in range(12, m) if seq_c[i] == seq_h[i])
        g = {'key': k, 'prefix_tokens': len(pre), 'agree_after_prefix': '%d/%d' % (agree, m - 12),
             'first_divergence': next((i for i in range(12, m) if seq_c[i] != seq_h[i]), None),
             'chain_text': tok.decode(seq_c), 'hf_text': tok.decode(seq_h)}
        gens.append(g)
        print('%s agree %s first divergence at %s\n   chain: %s\n   hf  : %s'
              % (k, g['agree_after_prefix'], g['first_divergence'],
                 repr(g['chain_text']), repr(g['hf_text'])), flush=True)
    out['greedy'] = gens
    p = args.out or os.path.join(ROOT, 'eval', 'sweep.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('WROTE %s' % p, flush=True)
    print('DONE_SWEEP', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ mode pad
if args.mode == 'pad':
    # Hypothesis this test decides. Nothing masks the unused half of the 512-slot cache: an all-zero key
    # scores exactly 0 against any query, so every empty slot carries attention weight 1.0 while the real
    # context carries exp(s). The damage is therefore exp(s)-shaped, not fixed -- flat, low-score contexts
    # (digits, short prompts) get diluted, peaky ones (natural CJK prose) do not. That is what the 60-item
    # run shows. Shrinking the cache shortens the pad, so if the mechanism is the pad mass, cos must climb
    # as SLOTS falls and the small-cache run must be able to beat the graph's own 512. The compiled ZG net
    # is frozen at 512, so a win here is an argument to recompile with a shorter context, not a fix in
    # itself. SLOTS is reassigned here and read by zero_cache()/e41_step().
    ids_by_item = {it['id']: it for it in json.load(open(args.items, encoding='utf-8'))}
    picks = [1, 5, 16, 17, 31, 34, 46, 51, 54, 55, 58, 60]
    ZH = ('今天早上下了小雨，街面上的水洼映着路灯。我撑着伞走过两个路口，卖早餐的摊子已经排起了队。'
          '摊主一边装豆浆，一边和熟客聊着这几天的天气。')
    EN = ('The morning was quiet, and a thin rain covered the street. Walking past the second crossing, '
          'I stopped at a small food stall where the owner was already serving the regulars.')
    cases = [('item%d' % i, tok(ids_by_item[i]['prompt'], return_tensors='pt',
                                add_special_tokens=False).input_ids[0].tolist()) for i in picks]
    cases.append(('zh24', tok(ZH, return_tensors='pt', add_special_tokens=False).input_ids[0][:24].tolist()))
    cases.append(('en24', tok(EN, return_tensors='pt', add_special_tokens=False).input_ids[0][:24].tolist()))
    grid = [512, 256, 128, 64, 32, 16]
    print('PAD cache-slot sweep, final-step logits vs HF float32 (grid %s)' % grid, flush=True)
    res = []
    for name, ids in cases:
        g = hf_full(hf32, ids)[-1]
        row = {'case': name, 'L': len(ids)}
        for slots in grid:
            if slots < len(ids) + 3:
                row['slots%d' % slots] = None
                continue
            SLOTS = slots
            lc = e41_chain(ids)[-1]
            # control: a cache of the same length but seeded with HF's own K/V for ids[:-1]. ORT-genai's
            # GroupQueryAttention may take the rope position from the past length rather than from the
            # token_dynamic_fm triple, in which case shortening the cache moves every position and the
            # comparison says nothing about pad mass. If this column stays flat while the chained column
            # climbs, the position is not coming from the cache length and the climb is the pad.
            ls = e41_seeded(ids)
            rc, rs = score(lc, g), score(ls, g)
            row['slots%d' % slots] = {'cos': rc['cos'], 'maxabs': rc['maxabs'],
                                      'rank_of_hf_top1': rc['rank_of_ref_top1'],
                                      'top1': int(lc.argmax()), 'same_as_hf': rc['top1_match'],
                                      'seeded_cos': rs['cos'], 'seeded_same_as_hf': rs['top1_match']}
            SLOTS = 512
        res.append(row)
        cells = ' '.join('%s:%s' % (s, row['slots%d' % s]['cos'] if row['slots%d' % s] else '-')
                         for s in grid)
        seed = ' '.join('%s:%s' % (s, row['slots%d' % s]['seeded_cos'])
                        for s in grid if row['slots%d' % s])
        hits = ' '.join('%s:%d' % (s, int(row['slots%d' % s]['same_as_hf']))
                        for s in grid if row['slots%d' % s])
        print('%-7s L=%2d chain cos %s\n        seeded cos %s\n        chain top1==hf %s'
              % (name, len(ids), cells, seed, hits), flush=True)
    outp = args.out or os.path.join(ROOT, 'eval', 'pad.json')
    json.dump(res, open(outp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for s in grid:
        vals = [r['slots%d' % s]['cos'] for r in res if r['slots%d' % s]]
        sv = [r['slots%d' % s]['seeded_cos'] for r in res if r['slots%d' % s]]
        ok = sum(1 for r in res if r['slots%d' % s] and r['slots%d' % s]['same_as_hf'])
        if vals:
            print('SLOTS %3d  chain mean cos %.5f (seeded control %.5f)  n=%d  chain top1==hf %d/%d'
                  % (s, sum(vals) / len(vals), sum(sv) / len(sv), len(vals), ok, len(vals)), flush=True)
    print('DONE_PAD', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ mode decomp
if args.mode == 'decomp':
    # A raw-logit maxabs of 20 alongside rank_of_hf_top1 = 0 cannot both be read at face value: log-softmax
    # is invariant to a constant added to every logit, so any part of the error that is a uniform shift
    # costs the model nothing. Split the error into shift, shape and probability, and report the parts the
    # sampler actually sees.
    items_all = {it['id']: it for it in json.load(open(args.items, encoding='utf-8'))}
    ZH = ('今天早上下了小雨，街面上的水洼映着路灯。我撑着伞走过两个路口，卖早餐的摊子已经排起了队。'
          '摊主一边装豆浆，一边和熟客聊着这几天的天气。')
    EN = ('The morning was quiet, and a thin rain covered the street. Walking past the second crossing, '
          'I stopped at a small food stall where the owner was already serving the regulars.')
    cases = [('item%d_%s' % (i, items_all[i]['prompt']), items_all[i]['prompt']) for i in (1, 5, 31, 54, 58)]
    cases += [('zh prose 33t', ZH), ('en prose 24t', EN)]
    print('DECOMP error split, chained net vs HF float32, final prompt position', flush=True)
    out = []
    for name, text in cases:
        ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
        g = hf_full(hf32, ids)[-1]
        lc = e41_chain(ids)[-1]
        ls = e41_seeded(ids)
        for tag, a in (('chain', lc), ('seeded', ls)):
            d = a - g
            mean = float(d.mean())
            dc = d - mean
            pg = np.exp(g - g.max()); pg /= pg.sum()
            pa = np.exp(a - a.max()); pa /= pa.sum()
            tv = float(0.5 * np.abs(pa - pg).sum())
            top = np.argsort(pg)[-5:][::-1]
            r = {'case': name, 'col': tag, 'L': len(ids),
                 'maxabs_raw': round(float(np.abs(d).max()), 4),
                 'shift': round(mean, 4),
                 'maxabs_after_shift': round(float(np.abs(dc).max()), 4),
                 'std_after_shift': round(float(dc.std()), 5),
                 'pearson': round(float(np.corrcoef(a, g)[0, 1]), 6),
                 'cos': round(float(a @ g / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-12)), 6),
                 'tv': round(tv, 6), 'kl': round(float((pg * np.log((pg + 1e-12) / (pa + 1e-12))).sum()), 5),
                 'p_hf_top1': round(float(pg[top[0]]), 5), 'p_col_of_hf_top1': round(float(pa[top[0]]), 5),
                 'top1_agree': bool(int(a.argmax()) == int(g.argmax())),
                 'hf_top5': [(int(t), tok.decode([int(t)]), round(float(pg[t]), 4),
                              round(float(pa[t]), 4)) for t in top]}
            out.append(r)
            print('%-22s %-7s L=%2d maxabs %7.3f (shift %7.3f, after %7.3f) pearson %.6f cos %.5f '
                  'TV %.5f KL %.4f | P(hf_top1) hf %s -> col %s | top1 %s'
                  % (name, tag, len(ids), r['maxabs_raw'], r['shift'], r['maxabs_after_shift'],
                     r['pearson'], r['cos'], r['tv'], r['kl'], r['p_hf_top1'],
                     r['p_col_of_hf_top1'], int(r['top1_agree'])), flush=True)
            print('     hf top5 %s' % r['hf_top5'], flush=True)
    p = args.out or os.path.join(ROOT, 'eval', 'decomp.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('WROTE %s' % p, flush=True)
    print('DONE_DECOMP', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ mode cachemap
if args.mode == 'cachemap':
    # Two different failures look identical in the logits. Either the chained walk writes each token's K/V
    # into the wrong slot (a positional bug the board driver could fix), or it writes them into the right
    # slot with the wrong content (rope at the wrong position, or fp16 drift the driver cannot fix). V is
    # the discriminator: values carry no rope, so a V slot can only be wrong if the hidden state that
    # produced it is wrong or it landed in the wrong place, while K can be wrong even when V is right.
    # So K and V are compared separately, and every chain slot is matched against *all* HF slots instead of
    # assuming offset 0 -- a shift shows up as a nonzero best offset, not as a low cos.
    #
    # The second question is whether the cache is write-once. In a correct decode graph step i writes slot i
    # and copies the rest through unchanged; if an already-written slot changes between steps, the graph is
    # recomputing the history and the drift is accumulation, not misplacement.
    items_all = {it['id']: it for it in json.load(open(args.items, encoding='utf-8'))}
    ZH = ('今天早上下了小雨，街面上的水洼映着路灯。我撑着伞走过两个路口，卖早餐的摊子已经排起了队。'
          '摊主一边装豆浆，一边和熟客聊着这几天的天气。')
    cases = [('ref7', tok.decode(REF_IDS)), ('item1', items_all[1]['prompt']),
             ('item31', items_all[31]['prompt']), ('item54', items_all[54]['prompt']),
             ('zh prose', ZH)]
    MUT = (0, 9, 17)          # layers whose slots get re-read after every step
    print('CACHEMAP chain cache vs HF cache, per slot and per layer', flush=True)
    out = []
    for name, text in cases:
        ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
        N = len(ids)
        with torch.no_grad():
            pkv = hf32(input_ids=torch.tensor([ids]), use_cache=True).past_key_values
        if hasattr(pkv, 'layers'):
            pairs = [(lyr.keys, lyr.values) for lyr in pkv.layers]
        else:
            pairs = list(zip(pkv[0], pkv[1]))
        hf_k = [np.repeat(k.float().numpy(), 4, axis=1) for k, v in pairs]
        hf_v = [np.repeat(v.float().numpy(), 4, axis=1) for k, v in pairs]

        cache = zero_cache()
        hist = {li: [] for li in MUT}
        for i, t in enumerate(ids):
            lg, upd = e41_step(cache, t, i + 1)
            cache.update(upd)
            for li in MUT:
                hist[li].append((cache['past_key_values.%d.key' % li][0, :, i + 1].astype(np.float32).copy(),
                                 cache['past_key_values.%d.value' % li][0, :, i + 1].astype(np.float32).copy()))
        final_slot = {}
        for li in MUT:
            for letter, which in (('key', 0), ('value', 1)):
                arr = cache['past_key_values.%d.%s' % (li, letter)][0].astype(np.float32)
                final_slot[(li, letter)] = arr
                drift = max(float(np.abs(arr[:, i + 1, :] - hist[li][i][which]).max()) for i in range(N))
                print('%-9s L=%2d %-5s slot drift after being written: maxabs %.6f'
                      % (name, li, letter, drift), flush=True)
        pad_max = max(float(np.abs(cache['past_key_values.%d.%s' % (li, le)][0, :, 0]).max())
                      for li in range(N_LAYER) for le in ('key', 'value'))
        print('%-9s slot0 maxabs over all 36 tensors: %.6f' % (name, pad_max), flush=True)

        def cmat(a, b):
            a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
            b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
            return a @ b.T

        def flat(x):
            return x.transpose(1, 0, 2).reshape(N, -1)   # [8,N,128] -> [N, 1024]

        rows = []
        for li in range(N_LAYER):
            r = {'case': name, 'layer': li}
            for letter, hf in (('key', hf_k), ('value', hf_v)):
                ch = cache['past_key_values.%d.%s' % (li, letter)][0, :, 1:N + 1].astype(np.float32)
                hp = flat(hf[li][0][:, :, PERM])
                hn = flat(hf[li][0])
                cf = flat(ch)
                cp = np.array([np.dot(cf[s], hp[s]) / (np.linalg.norm(cf[s]) * np.linalg.norm(hp[s]) + 1e-12)
                               for s in range(N)])
                cn = np.array([np.dot(cf[s], hn[s]) / (np.linalg.norm(cf[s]) * np.linalg.norm(hn[s]) + 1e-12)
                               for s in range(N)])
                # offset search runs in the basis the chain actually writes: K carries the rope permutation,
                # V provably does not (its nat cos is ~1 while its perm cos is ~0), so V searches natural.
                ref = hp if letter == 'key' else hn
                best = cmat(cf, ref)
                off = best.argmax(axis=1) - np.arange(N)
                r[letter + '_cos_perm'] = round(float(cp.mean()), 6)
                r[letter + '_cos_perm_min'] = round(float(cp.min()), 6)
                r[letter + '_cos_natural'] = round(float(cn.mean()), 6)
                r[letter + '_cos_nat_min'] = round(float(cn.min()), 6)
                r[letter + '_best_offset_mode'] = int(np.bincount(off + N, minlength=2 * N + 1).argmax()) - N
                r[letter + '_offset_zero_frac'] = round(float((off == 0).mean()), 4)
                r[letter + '_per_slot_cos'] = [round(float(x), 5) for x in (cp if letter == 'key' else cn)]
                r[letter + '_norm_ratio'] = round(float(np.linalg.norm(cf, axis=1).mean()
                                                        / np.linalg.norm(hp, axis=1).mean()), 5)
            rows.append(r)
            print('%-9s L=%2d | K perm %s nat %s off %s@%s ratio %s | V perm %s nat %s off %s@%s ratio %s'
                  % (name, li, r['key_cos_perm'], r['key_cos_natural'], r['key_best_offset_mode'],
                     r['key_offset_zero_frac'], r['key_norm_ratio'], r['value_cos_perm'],
                     r['value_cos_natural'], r['value_best_offset_mode'], r['value_offset_zero_frac'],
                     r['value_norm_ratio']), flush=True)
            if li in (0, 1, 2, 3, N_LAYER - 1):
                print('          slots K %s' % r['key_per_slot_cos'], flush=True)
                print('          slots V %s' % r['value_per_slot_cos'], flush=True)
        out.append({'case': name, 'tokens': N, 'ids': ids, 'slot0_maxabs': pad_max, 'layers': rows})
    p = args.out or os.path.join(ROOT, 'eval', 'cachemap.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('WROTE %s' % p, flush=True)
    print('DONE_CACHEMAP', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ mode padfix
if args.mode == 'padfix':
    # cachemap localised the chained drift to one slot: the entry the graph writes for token 0. At step 0 the
    # read window is slot 0 (all-zero pad) plus the token itself, so position 0's attention output is the
    # softmax over {pad, self} instead of the pure self-attention a prefill computes -- and every later step
    # then attends to that poisoned entry. The damage is a 1/(N+1) share, which is why 3-token prompts fall
    # apart (item1 slot0: K cos -0.29, V cos -0.27 at layer 2) while 30-token prose holds at 0.99.
    #
    # Repair needs no recompile, only a driver convention: after the first step, copy slot 1 into slot 0 and
    # feed token 0 again. Now the window is {self, self} -- two identical keys and values, whose softmax is
    # exactly self-attention -- so position 0 is computed correctly. The residue is that token 0 then carries
    # double weight for the rest of the walk, which is a mild over-weighting rather than a zero-value leak.
    def chain_run(ids, repair):
        """Walk the prompt one token at a time; with repair=True use the slot0=slot1 double first step.

        Returns (logits, cache) so the greedy continuation can pick up where the prompt left off.
        """
        cache = zero_cache()
        logits = []

        def seed_pad():
            for nm in cache:
                cache[nm][0, :, 0, :] = cache[nm][0, :, 1, :]

        if repair:
            lg, upd = e41_step(cache, ids[0], 1)
            cache.update(upd)
            seed_pad()
            lg, upd = e41_step(cache, ids[0], 1)
            logits.append(lg)
            cache.update(upd)
            seed_pad()
            rng = range(1, len(ids))
        else:
            rng = range(len(ids))
        for i in rng:
            lg, upd = e41_step(cache, ids[i], i + 1)
            logits.append(lg)
            cache.update(upd)
        return logits, cache

    def chain_fixed(ids):
        return chain_run(ids, True)[0]

    def greedy(pre, steps, repair):
        seq = list(pre)
        logits, cache = chain_run(seq, repair)
        for _ in range(steps):
            seq.append(int(logits[-1].argmax()))
            lg, upd = e41_step(cache, seq[-1], len(seq))
            logits.append(lg)
            cache.update(upd)
        return seq

    items_all = {it['id']: it for it in json.load(open(args.items, encoding='utf-8'))}
    ZH = ('今天早上下了小雨，街面上的水洼映着路灯。我撑着伞走过两个路口，卖早餐的摊子已经排起了队。'
          '摊主一边装豆浆，一边和熟客聊着这几天的天气。')
    EN = ('The morning was quiet, and a thin rain covered the street. Walking past the second crossing, '
          'I stopped at a small food stall where the owner was already serving the regulars.')
    print('PADFIX plain chain vs pad-repaired chain, both against HF float32', flush=True)
    recs = []
    cols = {}
    for tag in ('e41_chain', 'e41_fixed'):
        cols[tag] = {'cos': [], 'tv': [], 'maxabs': [], 'agree': 0, 'acc': 0, 'rank': []}
    for it in json.load(open(args.items, encoding='utf-8')):
        ids = tok(as_prompt(it['prompt']), return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
        tgt = tok(it['answer'], return_tensors='pt', add_special_tokens=False).input_ids[0][0]
        g = hf_full(hf32, ids)[-1]
        pg = np.exp(g - g.max()); pg /= pg.sum()
        runs = {'e41_chain': e41_chain(ids)[-1], 'e41_fixed': chain_fixed(ids)[-1]}
        line = '[%2d] %-8s L=%2d %-14s' % (it['id'], it['cat'], len(ids), repr(it['prompt'])[:14])
        for tag, a in runs.items():
            pa = np.exp(a - a.max()); pa /= pa.sum()
            c = cols[tag]
            c['cos'].append(float(a @ g / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-12)))
            c['tv'].append(float(0.5 * np.abs(pa - pg).sum()))
            c['maxabs'].append(float(np.abs(a - g).max()))
            c['agree'] += int(int(a.argmax()) == int(g.argmax()))
            c['acc'] += int(int(a.argmax()) == int(tgt))
            c['rank'].append(int((a > a[tgt]).sum()))
            line += ' | %s cos %.5f TV %.4f top1 %s' % (tag[-5:], c['cos'][-1], c['tv'][-1],
                                                        int(a.argmax()) == int(g.argmax()))
        recs.append({'id': it['id'], 'cat': it['cat'], 'L': len(ids), 'hf_top1': int(g.argmax()),
                     'chain_top1': int(runs['e41_chain'].argmax()),
                     'fixed_top1': int(runs['e41_fixed'].argmax()),
                     'chain_cos': round(cols['e41_chain']['cos'][-1], 6),
                     'fixed_cos': round(cols['e41_fixed']['cos'][-1], 6),
                     'chain_tv': round(cols['e41_chain']['tv'][-1], 6),
                     'fixed_tv': round(cols['e41_fixed']['tv'][-1], 6)})
        print(line, flush=True)
    for text, tag in ((ZH, 'zh prose'), (EN, 'en prose')):
        ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
        g = hf_full(hf32, ids)[-1]
        for name, lg in (('e41_chain', e41_chain(ids)[-1]), ('e41_fixed', chain_fixed(ids)[-1])):
            pg = np.exp(g - g.max()); pg /= pg.sum()
            pa = np.exp(lg - lg.max()); pa /= pa.sum()
            print('%-9s %-11s L=%2d cos %.5f TV %.4f maxabs %.3f top1 %s'
                  % (tag, name, len(ids), lg @ g / (np.linalg.norm(lg) * np.linalg.norm(g) + 1e-12),
                     0.5 * np.abs(pa - pg).sum(), np.abs(lg - g).max(),
                     int(lg.argmax()) == int(g.argmax())), flush=True)
    n = len(recs)
    print('PADFIX n=%d | chain: hf-top1 agree %d/%d  acc %d/%d  cos %.5f  TV %.4f  rank %.1f'
          % (n, cols['e41_chain']['agree'], n, cols['e41_chain']['acc'], n,
             sum(cols['e41_chain']['cos']) / n, sum(cols['e41_chain']['tv']) / n,
             sum(cols['e41_chain']['rank']) / n), flush=True)
    print('        fixed: hf-top1 agree %d/%d  acc %d/%d  cos %.5f  TV %.4f  rank %.1f'
          % (cols['e41_fixed']['agree'], n, cols['e41_fixed']['acc'], n,
             sum(cols['e41_fixed']['cos']) / n, sum(cols['e41_fixed']['tv']) / n,
             sum(cols['e41_fixed']['rank']) / n), flush=True)
    for lo, hi in ((1, 3), (4, 5), (6, 8), (9, 100)):
        sel = [r for r in recs if lo <= r['L'] <= hi]
        if not sel:
            continue
        print('LEN %2d-%2d n=%2d | chain agree %.2f cos %.5f TV %.4f | fixed agree %.2f cos %.5f TV %.4f'
              % (lo, hi, len(sel),
                 sum(1 for r in sel if r['chain_top1'] == r['hf_top1']) / len(sel),
                 sum(r['chain_cos'] for r in sel) / len(sel),
                 sum(r['chain_tv'] for r in sel) / len(sel),
                 sum(1 for r in sel if r['fixed_top1'] == r['hf_top1']) / len(sel),
                 sum(r['fixed_cos'] for r in sel) / len(sel),
                 sum(r['fixed_tv'] for r in sel) / len(sel)), flush=True)
    p = args.out or os.path.join(ROOT, 'eval', 'padfix.json')
    hurt = [r for r in recs if r['fixed_tv'] > r['chain_tv']]
    print('REPAIR HURTS on %d/%d: %s' % (len(hurt), n, [(r['id'], r['L']) for r in hurt]), flush=True)

    # Free-running generation is the condition the device actually runs in: the walk feeds its own argmax, so
    # a per-step defect compounds. Compare plain chain, repaired chain and HF greedy on the same 12-token prefix.
    print('GREEDY free-running continuation, 24 tokens', flush=True)
    gens = []
    for it in (items_all[1], items_all[31], items_all[58]):
        pre = tok(as_prompt(it['prompt']), return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
        if not args.chat:
            pre = pre[:12]
        seqs = {'chain': greedy(pre, 24, False), 'fixed': greedy(pre, 24, True)}
        seq_h = list(pre)
        for _ in range(24):
            seq_h.append(int(hf_full(hf32, seq_h)[-1].argmax()))
        seqs['hf'] = seq_h
        g = {'key': 'item%d_%s' % (it['id'], it['prompt']), 'prefix': pre}
        for tag in ('chain', 'fixed'):
            m = min(len(seqs[tag]), len(seq_h))
            g[tag] = {'agree_vs_hf': '%d/%d' % (sum(1 for i in range(len(pre), m)
                                                    if seqs[tag][i] == seq_h[i]), m - len(pre)),
                      'first_divergence': next((i for i in range(len(pre), m)
                                                if seqs[tag][i] != seq_h[i]), None),
                      'text': tok.decode(seqs[tag])}
        g['hf_text'] = tok.decode(seq_h)
        gens.append(g)
        print('%s\n   chain %s div@%s: %s\n   fixed %s div@%s: %s\n   hf   : %s'
              % (g['key'], g['chain']['agree_vs_hf'], g['chain']['first_divergence'],
                 repr(g['chain']['text']), g['fixed']['agree_vs_hf'], g['fixed']['first_divergence'],
                 repr(g['fixed']['text']), repr(g['hf_text'])), flush=True)
    json.dump({'items': recs, 'greedy': gens}, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('WROTE %s' % p, flush=True)
    print('DONE_PADFIX', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ mode items
items = json.load(open(args.items, encoding='utf-8'))
if args.limit:
    items = items[:args.limit]
print('ITEMS %d' % len(items), flush=True)

recs, t_all = [], time.time()
for it in items:
    ids = tok(as_prompt(it['prompt']), return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
    ans = tok(it['answer'], return_tensors='pt', add_special_tokens=False).input_ids[0].tolist()
    tgt = ans[0]
    g_all = hf_full(hf32, ids)
    g = g_all[-1]
    cols = {}
    cols['hf16'] = hf_full(hf16, ids)[-1]
    cols['native'] = native_full(ids)
    cols['e41_seeded'] = e41_seeded(ids)
    chain_steps = e41_chain(ids)
    cols['e41_chain'] = chain_steps[-1]
    rec = {'id': it['id'], 'cat': it['cat'], 'prompt': it['prompt'], 'answer': it['answer'],
           'n_tokens': len(ids), 'answer_tokens': ans[:4], 'target': tgt,
           'hf32_top1': int(g.argmax()), 'hf32_prob': None, 'target_rank_hf32': int((g > g[tgt]).sum())}
    pa = np.exp(g - g.max()); pa /= pa.sum()
    rec['hf32_prob'] = round(float(pa[tgt]), 5)
    for name, lg in cols.items():
        r = score(lg, g)
        r['top1'] = int(lg.argmax())
        r['target_rank'] = int((lg > lg[tgt]).sum())
        r['acc_top1'] = bool(r['top1'] == tgt)
        rec[name] = r
    # how many of the prompt's own positions the chained walk still agrees with HF on, and how far down
    # the last step has drifted: this is what the board would show while digesting a long prompt
    rec['chain_step_agree'] = sum(1 for i in range(len(ids))
                                  if int(chain_steps[i].argmax()) == int(g_all[i].argmax()))
    rec['chain_step_cos_last'] = rec['e41_chain']['cos']
    recs.append(rec)
    print('[%2d] %-8s L=%2d %-22s hf32->%d tgt_rank=%2d | seeded cos %s top1 %d rank %d | '
          'chain cos %s top1 %d rank %d | native cos %s | hf16 cos %s'
          % (it['id'], it['cat'], len(ids), repr(it['prompt'])[:22], rec['hf32_top1'],
             rec['target_rank_hf32'], rec['e41_seeded']['cos'], rec['e41_seeded']['top1'],
             rec['e41_seeded']['target_rank'], rec['e41_chain']['cos'], rec['e41_chain']['top1'],
             rec['e41_chain']['target_rank'], rec['native']['cos'], rec['hf16']['cos']), flush=True)
print('ITEMS done in %.1f s' % (time.time() - t_all), flush=True)


def agg(rows, name):
    if name == 'hf32':
        return {'n': len(rows),
                'acc_top1': round(sum(1 for r in rows if r['target_rank_hf32'] == 0) / len(rows), 4),
                'mean_target_rank': round(sum(r['target_rank_hf32'] for r in rows) / len(rows), 2),
                'mean_prob': round(sum(r['hf32_prob'] for r in rows) / len(rows), 5),
                'cos': 1.0, 'maxabs': 0.0, 'kl': 0.0, 'top1_match_vs_hf32': 1.0}
    s = [r[name] for r in rows]
    return {'n': len(s),
            'acc_top1': round(sum(1 for x in s if x['acc_top1']) / len(s), 4),
            'mean_target_rank': round(sum(x['target_rank'] for x in s) / len(s), 2),
            'cos': round(sum(x['cos'] for x in s) / len(s), 6),
            'cos_min': round(min(x['cos'] for x in s), 6),
            'maxabs': round(sum(x['maxabs'] for x in s) / len(s), 4),
            'maxabs_max': round(max(x['maxabs'] for x in s), 4),
            'kl': round(sum(x['kl'] for x in s) / len(s), 5),
            'kl_max': round(max(x['kl'] for x in s), 5),
            'top1_match_vs_hf32': round(sum(1 for x in s if x['top1_match']) / len(s), 4),
            'mean_rank_of_hf32_top1': round(sum(x['rank_of_ref_top1'] for x in s) / len(s), 2)}


summary = {'overall': {n: agg(recs, n) for n in ('hf32', 'hf16', 'native', 'e41_seeded', 'e41_chain')}}
summary['overall']['e41_chain']['mean_step_agree'] = round(
    sum(r['chain_step_agree'] for r in recs) / len(recs), 2)
summary['overall']['e41_chain']['mean_prompt_len'] = round(
    sum(r['n_tokens'] for r in recs) / len(recs), 2)
cats = sorted({r['cat'] for r in recs})
summary['by_cat'] = {c: {n: agg([r for r in recs if r['cat'] == c], n)
                         for n in ('hf32', 'native', 'e41_chain')} for c in cats}
for c in cats:
    print('CAT %-10s hf32 acc %s | e41_chain acc %s cos %s top1match %s'
          % (c, summary['by_cat'][c]['hf32']['acc_top1'],
             summary['by_cat'][c]['e41_chain']['acc_top1'],
             summary['by_cat'][c]['e41_chain']['cos'],
             summary['by_cat'][c]['e41_chain']['top1_match_vs_hf32']), flush=True)
for n in ('hf16', 'native', 'e41_seeded', 'e41_chain'):
    print('AGG %-11s acc %s rank %s cos %s (min %s) maxabs %s (max %s) kl %s top1match %s hf1rank %s'
          % (n, summary['overall'][n]['acc_top1'], summary['overall'][n]['mean_target_rank'],
             summary['overall'][n]['cos'], summary['overall'][n].get('cos_min'),
             summary['overall'][n]['maxabs'], summary['overall'][n].get('maxabs_max'),
             summary['overall'][n]['kl'], summary['overall'][n]['top1_match_vs_hf32'],
             summary['overall'][n].get('mean_rank_of_hf32_top1')), flush=True)
print('AGG hf32(ceiling) acc %s rank %s prob %s'
      % (summary['overall']['hf32']['acc_top1'], summary['overall']['hf32']['mean_target_rank'],
         summary['overall']['hf32']['mean_prob']), flush=True)

out = args.out or os.path.join(ROOT, 'eval', 'eval_items.json')
json.dump({'summary': summary, 'items': recs}, open(out, 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
print('WROTE %s' % out, flush=True)
print('DONE_ITEMS', flush=True)
