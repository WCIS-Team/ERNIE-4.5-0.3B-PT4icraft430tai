"""Print builders/ernie.py and the HF ernie4_5 rope lines with real file paths and numbers."""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import onnxruntime_genai.models.builder as B  # side effect: puts builders/ on sys.path
import builders.ernie as E
import transformers.models.ernie4_5.modeling_ernie4_5 as H

print('ernie_module=%s' % E.__file__)
print('hf_module=%s' % H.__file__)
for mod, rngs in ((E, [range(1, 46)]), (H, [range(100, 120), range(166, 196)])):
    lines = open(mod.__file__, encoding='utf-8', errors='replace').read().splitlines()
    print('=== %s (%d lines) ===' % (mod.__file__, len(lines)))
    for rng in rngs:
        for i in rng:
            if 0 <= i - 1 < len(lines):
                print('%4d|%s' % (i, lines[i - 1][:165]))
