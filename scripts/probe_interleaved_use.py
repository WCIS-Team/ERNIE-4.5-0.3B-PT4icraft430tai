"""Where does rope_attrs['interleaved'] get consumed, and is any weight permuted to match?"""
import inspect
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import onnxruntime_genai.models.builder as B
import builders.ernie as E

files = {}
for k in E.ErnieModel.__mro__:
    try:
        files[k.__name__] = inspect.getsourcefile(k)
    except Exception:
        pass
print('MRO files=%s' % files)

for cls, f in files.items():
    if f is None:
        continue
    src = open(f, encoding='utf-8', errors='replace').read()
    hits = [(i + 1, l.strip()[:165]) for i, l in enumerate(src.splitlines()) if 'interleaved' in l]
    print('=== %s (%s) interleaved-hits=%d ===' % (cls, f, len(hits)))
    for n, l in hits[:20]:
        print('%4d|%s' % (n, l))

f = files.get('Model')
if f:
    src = open(f, encoding='utf-8', errors='replace').read()
    for name in ('def make_rope_init', 'def create_rotary_embedding_caches', 'def make_rotary_embedding_caches'):
        g = re.search(re.escape(name) + r'.*?(?=\n    def )', src, re.S)
        if g:
            print('=== %s ===' % name)
            for i, l in enumerate(g.group(0).splitlines()[:40]):
                print('%3d|%s' % (i, l[:165]))
