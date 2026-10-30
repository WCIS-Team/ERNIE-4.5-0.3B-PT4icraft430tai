"""Locate the module that defines genai's ErnieModel and read its RoPE / interleaved handling."""
import inspect
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import onnxruntime_genai.models.builder as B

src = inspect.getsource(B)
print('=== import block mentioning Ernie ===')
for i, l in enumerate(src.splitlines()):
    if 'Ernie' in l or ('import' in l and 'model' in l.lower()):
        print('%4d|%s' % (i, l.strip()[:170]))

# resolve the symbol
cls = getattr(B, 'ErnieModel', None)
print('ErnieModel in builder ns=%r' % (cls,))
if cls is None:
    # find candidate modules already imported by builder
    mods = sorted({m for m in sys.modules if 'onnxruntime_genai' in m})
    print('genai modules loaded: %s' % mods)
    for name in mods:
        c = getattr(sys.modules[name], 'ErnieModel', None)
        if c is not None:
            print('FOUND ErnieModel in %s (file=%s)' % (name, getattr(sys.modules[name], '__file__', '?')))
            cls = c
            break
if cls is not None:
    print('=== MRO ===')
    print(' -> '.join(k.__name__ for k in cls.__mro__))
    for k in cls.__mro__:
        if k.__name__ in ('TransformerModelBuilder', 'QwenModelBuilder', 'BaseModelBuilder'):
            f = inspect.getsourcefile(k)
            print('base %s file=%s' % (k.__name__, f))
    text = ''
    for k in cls.__mro__:
        try:
            text += inspect.getsource(k)
        except Exception as e:
            print('getsource failed for %s: %s' % (k.__name__, e))
    print('combined_source_lines=%d' % len(text.splitlines()))
    for kw in ('interleaved', 'rotate', 'rotary', 'position_ids_base', 'rope'):
        hits = [l.strip()[:160] for l in text.splitlines() if kw in l]
        print('-- %s: %d' % (kw, len(hits)))
        for h in hits[:20]:
            print('   |%s' % h)
