"""Probe whether the installed genai builder and transformers know Ernie4_5."""
import inspect
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import onnxruntime_genai.models.builder as B

src = inspect.getsource(B)
print('builder_file=%s' % B.__file__)
print('builder_lines=%d' % len(src.splitlines()))
pat = re.compile(r'^.*[Ee]rnie.*$', re.M)
hits = pat.findall(src)
print('ernie_hits=%d' % len(hits))
for h in hits[:40]:
    print('  |%s' % h.strip()[:150])

print('=== model_type dispatch ===')
for name in ('MODEL_TYPE_TO_CLASS', 'model_type_to_class', 'MODEL_TYPE_MAPPING'):
    obj = getattr(B, name, None)
    if obj:
        print('%s=%s' % (name, sorted(obj)))
low = [n for n in dir(B) if 'uilder' in n]
print('builder_classes=%s' % low[:40])

print('=== transformers side ===')
import transformers
print('transformers=%s' % transformers.__version__)
for cand in ('Ernie4_5Model', 'ErnieModel', 'Ernie4_5ForCausalLM', 'Ernie4_5Config'):
    print('%-22s %s' % (cand, hasattr(transformers, cand)))
mods = [m for m in transformers.models.__all__ if 'ernie' in m.lower()]
print('ernie_model_modules=%s' % mods)
