"""Dump genai's ErnieModel builder source + how it sets RoPE interleaving, and the HF side."""
import inspect
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import onnxruntime_genai.models.builder as B

src = inspect.getsource(B)

m = re.search(r'class ErnieModel.*?(?=\nclass )', src, re.S)
print('=== builder ErnieModel (%d lines) ===' % (0 if not m else len(m.group(0).splitlines())))
if m:
    body = m.group(0)
    for i, l in enumerate(body.splitlines()):
        print('%3d|%s' % (i, l[:170]))

print('=== interleaved / rotary in whole builder ===')
for kw in ('interleaved', 'rotary', 'rotate_half', 'extra_options.get'):
    hits = [l.strip()[:150] for l in src.splitlines() if kw in l]
    print('-- %s: %d' % (kw, len(hits)))
    for h in hits[:18]:
        print('   |%s' % h)

print('=== HF modeling_ernie4_5 rope ===')
import transformers.models.ernie4_5.modeling_ernie4_5 as M
hsrc = inspect.getsource(M)
print('hf_lines=%d' % len(hsrc.splitlines()))
for kw in ('rotate_half', 'interleav', 'def apply_rotary_pos_emb', 'cos', 'indices'):
    hits = [l.strip()[:160] for l in hsrc.splitlines() if kw in l]
    print('-- %s: %d' % (kw, len(hits)))
    for h in hits[:8]:
        print('   |%s' % h)
g = re.search(r'def apply_rotary_pos_emb.*?(?=\ndef |\nclass )', hsrc, re.S)
if g:
    print('--- apply_rotary_pos_emb ---')
    for i, l in enumerate(g.group(0).splitlines()[:30]):
        print('%3d|%s' % (i, l[:170]))
g2 = re.search(r'def rotate_half.*?(?=\ndef |\nclass )', hsrc, re.S)
if g2:
    print('--- rotate_half ---')
    print(g2.group(0)[:600])
