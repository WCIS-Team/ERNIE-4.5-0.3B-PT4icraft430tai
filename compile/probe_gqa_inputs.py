"""Dump the producer chain of every GQA input slot for one fmodel dir.

replicate_qkv_columns.py assumed slot 0 is fed directly by the qkv Split. After add_head_roundtrip.py
that is no longer true, so this walks each slot back to a MatMul or an initializer and prints the
Split's sizes operand plus every layer-0 attention initializer name.
"""
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def prod_of(g, name):
    for n in g.node:
        if name in list(n.output):
            return n
    return None


def main():
    d = os.path.join(ROOT, sys.argv[1])
    m = onnx.load(os.path.join(d, 'model.onnx'), load_external_data=False)
    g = m.graph
    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    print('nodes %d; GQA %d; inputs %d outputs %d'
          % (len(g.node), len(gqas), len(g.input), len(g.output)))
    n0 = gqas[0]
    print('GQA0 name=%s attrs=%s' % (n0.name, {a.name: (a.i if a.i else a.f) for a in n0.attribute}))
    for s, ip in enumerate(n0.input):
        print('--- slot %d : %s' % (s, ip))
        cur = ip
        for depth in range(10):
            p = prod_of(g, cur)
            if p is None:
                for i in g.initializer:
                    if i.name == cur:
                        print('    [%d] INITIALIZER %s dims=%s' % (depth, cur, list(i.dims)))
                        break
                else:
                    for vi in list(g.value_info) + list(g.input):
                        if vi.name == cur:
                            print('    [%d] GRAPH INPUT/VALUE_INFO %s shape=%s'
                                  % (depth, cur, [str(x.dim_value or x.dim_param)
                                                  for x in vi.type.tensor_type.shape.dim]))
                            break
                    else:
                        print('    [%d] DANGLING %s' % (depth, cur))
                break
            print('    [%d] %s %s out=%s in=%s'
                  % (depth, p.op_type, p.name, list(p.output), list(p.input)))
            if p.op_type == 'Split':
                sizes = prod_of(g, p.input[1])
                if sizes is not None and sizes.op_type == 'Constant':
                    print('        sizes =', onnx.numpy_helper.to_array(sizes.value).tolist())
                else:
                    print('        sizes operand =', p.input[1])
                if sizes is None:
                    for i in g.initializer:
                        if i.name == p.input[1]:
                            print('        sizes =', onnx.numpy_helper.to_array(i).tolist())
            cur = p.input[0] if p.input else ''
            if not cur:
                break
    print('=== layer0 attention initializers ===')
    for i in g.initializer:
        if 'layers.0' in i.name and 'attn' in i.name:
            print('%-70s %s' % (i.name, list(i.dims)))
    print('=== layer0 attention nodes ===')
    for n in g.node:
        if 'layers.0' in n.name and 'attn' in n.name:
            print('%-40s %-70s %s' % (n.op_type, n.name, list(n.input)))


if __name__ == '__main__':
    main()
