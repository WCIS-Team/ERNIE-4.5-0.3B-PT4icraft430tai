"""Write a loadable twin of the rewritten graph so onnxruntime can run it as the pre-compile control.

e_v35 refuses to load in ORT-genai: `com.microsoft:SimplifiedLayerNormalization(-1) is not a registered
function/op`. Two metadata choices in this pipeline are responsible, both made for icraft's parser --
SimplifiedLayerNormalization was given the com.microsoft domain, and the graph was pulled down to
ir_version 8 / opset 17. The export itself carries the op in the default domain at opset 22.

Nothing else is touched: the Split/Concat kv 2->8 tiling, the shrunken rope cache, rotary_interleaved=0 and
every weight byte stay exactly as the compile input has them. So if this twin reproduces the compiled net's
error against the HF golden, the rewrites own it and icraft added nothing.

Saved next to the original in the same directory, so its external-data reference (model.onnx.data) still
resolves; the data file itself is not rewritten.

Usage: python -X utf8 mk_ort_variant.py --dir e_v35
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True)
ap.add_argument('--out', default='model_ort.onnx')
ap.add_argument('--ri', type=int, default=None,
                help='set rotary_interleaved on every GroupQueryAttention. The pipeline carries 0 '
                     '(patch_attr.py flipped it as a parser diagnostic and it was never restored), '
                     'while the cos/sin bytes are still the exporter''s interleaved layout')
args = ap.parse_args()

d = os.path.join(ROOT, args.dir)
m = onnx.load(os.path.join(d, 'model.onnx'), load_external_data=False)
g = m.graph
before = [(n.domain, n.op_type) for n in g.node if n.op_type == 'SimplifiedLayerNormalization']
moved = 0
for n in g.node:
    if n.op_type == 'SimplifiedLayerNormalization' and n.domain == 'com.microsoft':
        n.domain = ''
        moved += 1
m.ir_version = 10
for o in m.opset_import:
    if o.domain == '':
        print('default opset %d -> 22' % o.version)
        o.version = 22
print('%d SLN nodes, %d moved back to the default domain, %d left in com.microsoft'
      % (len(before), moved, len(before) - moved))
if args.ri is not None:
    seen = set()
    for n in g.node:
        if n.op_type == 'GroupQueryAttention':
            for a in n.attribute:
                if a.name == 'rotary_interleaved':
                    seen.add(a.i)
                    a.i = int(args.ri)
    print('rotary_interleaved: was %s on every GQA -> %d (%d nodes)'
          % (sorted(seen), args.ri, sum(1 for n in g.node if n.op_type == 'GroupQueryAttention')))
print('nodes %d inits %d inputs %d outputs %d'
      % (len(g.node), len(g.initializer), len(g.input), len(g.output)))
assert len(g.node) == 303, 'topology changed'
p = os.path.join(d, args.out)
onnx.save(m, p)
print('wrote %s (%d bytes); data file untouched (%d bytes)'
      % (p, os.path.getsize(p), os.path.getsize(os.path.join(d, 'model.onnx.data'))))
print('DONE_ORT_VARIANT')
