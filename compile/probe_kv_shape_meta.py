"""Print value_info / graph input-output shapes for every layer-0 attention tensor, plus GQA attrs."""
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def shape(vi):
    return [('%s' % (d.dim_value if d.HasField('dim_value') else d.dim_param))
            for d in vi.type.tensor_type.shape.dim]


def main():
    m = onnx.load(os.path.join(ROOT, sys.argv[1], 'model.onnx'), load_external_data=False)
    g = m.graph
    print('== graph inputs ==')
    for v in g.input:
        if 'layers.0' in v.name or v.name in ('past_key_values.0.key', 'past_key_values.0.value'):
            print('%-58s %s' % (v.name, shape(v)))
    print('== graph outputs ==')
    for v in g.output:
        print('%-58s %s' % (v.name, shape(v)))
    print('== value_info mentioning layers.0 attn ==')
    for v in g.value_info:
        n = v.name
        if ('layers.0' in n and 'attn' in n) or 'q_norm' in n or 'k_norm' in n:
            print('%-72s %s' % (n, shape(v)))
    print('== GQA node outputs in value_info ==')
    for node in g.node:
        if node.op_type == 'GroupQueryAttention':
            for o in node.output:
                hit = [v for v in list(g.value_info) + list(g.output) if v.name == o]
                print('%-58s -> %s' % (o, shape(hit[0]) if hit else 'NO SHAPE'))
            break
    print('== distinct /model/constants/INT64 values (count) ==')
    from collections import Counter
    c = Counter()
    for n in g.node:
        if n.op_type == 'Constant' and '/model/constants/INT64/' in n.name:
            a = onnx.numpy_helper.to_array(n.value)
            if a.dtype == np.int64 and a.size <= 4:
                c[(', '.join(str(x) for x in a.reshape(-1).tolist()), tuple(a.shape))] += 1
    for k in sorted(c, key=str):
        print('%-30s x%d' % (k, c[k]))
    print('== every int attr of every GQA, all layers ==')
    seen = {}
    for node in g.node:
        if node.op_type == 'GroupQueryAttention':
            for a in node.attribute:
                val = a.i if a.i else a.f
                seen.setdefault(a.name, set()).add(val)
    print({k: sorted(v, key=str) for k, v in seen.items()})


if __name__ == '__main__':
    main()
