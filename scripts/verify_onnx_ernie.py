"""Numeric check: exported genai ONNX vs Hugging Face golden, prefill + one decode step.

Adapted from repos/Qwen3-0.6B-Instruct/compile/verify_onnx.py; only SRC/ONNX/PROMPT differ.
Golden is recomputed with use_cache=False over the full sequence, so the decode step
tests that the ONNX graph's own KV cache is arithmetically continuous.
"""
import json, os, sys
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = r'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT'
ONNX_DIR = sys.argv[1] if len(sys.argv) > 1 else r'D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda'
ONNX = os.path.join(ONNX_DIR, 'model.onnx')
PROMPT = '请用一句话介绍量子计算。'
PROVS = (sys.argv[2].split(',') if len(sys.argv) > 2 else ['CPUExecutionProvider'])

cfg = json.load(open(os.path.join(SRC, 'config.json'), encoding='utf-8'))
n_layers = cfg['num_hidden_layers']
n_kv = cfg['num_key_value_heads']
head_dim = cfg.get('head_dim') or cfg['hidden_size'] // cfg['num_attention_heads']
print('CFG layers=%d heads=%d kv=%d head_dim=%d hidden=%d vocab=%d tie=%s' % (
    n_layers, cfg['num_attention_heads'], n_kv, head_dim, cfg['hidden_size'], cfg['vocab_size'],
    cfg.get('tie_word_embeddings')))

tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32).eval()
ids = tok(PROMPT, return_tensors='pt').input_ids
print('PROMPT_TOKENS', ids.shape[1], ids[0][:8].tolist())


def golden(token_ids):
    with torch.no_grad():
        out = model(input_ids=token_ids, use_cache=False)
    return out.logits[0, -1].float().numpy()


gold_last = golden(ids)
gold_tok = int(gold_last.argmax())
print('GOLD top5', np.argsort(-gold_last)[:5].tolist(), 'next', gold_tok)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(ONNX, so, providers=PROVS)
print('ORT_PROVIDERS_USED', sess.get_providers())
byname = {i.name: i for i in sess.get_inputs()}
onames = [o.name for o in sess.get_outputs()]
print('ONNX_INPUTS', len(sess.get_inputs()), 'ONNX_OUTPUTS', len(sess.get_outputs()))

kv_type = byname['past_key_values.0.key'].type
kv_dt = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[kv_type]
print('KV_DTYPE', kv_type)


def report(tag, onnx_logits, gold):
    a = onnx_logits.astype(np.float64)
    b = gold.astype(np.float64)
    d = np.abs(a - b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print('%s maxabs=%.5f relmax=%.5f cos=%.6f' % (tag, d.max(), d.max() / (np.abs(b).max() + 1e-9), cos))
    print('%s top5_onnx %s next %d | top5_gold %s next %d | MATCH_TOP1 %s' % (
        tag, np.argsort(-a)[:5].tolist(), int(a.argmax()), np.argsort(-b)[:5].tolist(), int(b.argmax()),
        int(a.argmax()) == int(b.argmax())))


S = int(ids.shape[1])
feeds = {
    'input_ids': ids.numpy().astype(np.int64),
    'attention_mask': np.ones((1, S), dtype=np.int64),
}
if 'position_ids' in byname:
    feeds['position_ids'] = np.arange(S, dtype=np.int64).reshape(1, S)
for i in range(n_layers):
    for tag in ('key', 'value'):
        feeds['past_key_values.%d.%s' % (i, tag)] = np.zeros((1, n_kv, 0, head_dim), dtype=kv_dt)
res = dict(zip(onames, sess.run(None, feeds)))
report('PREFILL', res['logits'][0, -1], gold_last)

next_id = np.array([[gold_tok]], dtype=np.int64)
feeds2 = {
    'input_ids': next_id,
    'attention_mask': np.ones((1, S + 1), dtype=np.int64),
}
if 'position_ids' in feeds:
    feeds2['position_ids'] = np.arange(S, S + 1, dtype=np.int64).reshape(1, 1)
for i in range(n_layers):
    for tag in ('key', 'value'):
        feeds2['past_key_values.%d.%s' % (i, tag)] = res['present.%d.%s' % (i, tag)]
r2 = dict(zip(onames, sess.run(None, feeds2)))
report('DECODE', r2['logits'][0, -1], golden(torch.cat([ids, torch.tensor([[gold_tok]])], dim=1)))
print('DONE_VERIFY')
