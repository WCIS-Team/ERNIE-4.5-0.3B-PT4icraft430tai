"""Census the exported ERNIE genai ONNX: metadata, IO, op histogram, rope/attention attrs.

Usage: python census_onnx.py <dir-with-model.onnx> [<second-dir-for-comparison>]
"""
import collections
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def census(path, tag):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    print('=== %s ===' % tag)
    print('ir=%d opset=%s producer=%s/%s' % (
        m.ir_version,
        [(oi.domain or 'ai.onnx', oi.version) for oi in m.opset_import],
        m.producer_name, m.producer_version))
    print('nodes=%d initializers=%d inputs=%d outputs=%d' % (
        len(g.node), len(g.initializer), len(g.input), len(g.output)))

    def dims(vi):
        t = vi.type.tensor_type
        out = []
        for d in t.shape.dim:
            out.append(d.dim_value if d.HasField('dim_value') else d.dim_param)
        return '%s%s' % (onnx.TensorProto.DataType.Name(t.elem_type), out)

    print('--- inputs ---')
    for i, vi in enumerate(g.input):
        if i < 4 or i >= len(g.input) - 2:
            print('  [%2d] %-28s %s' % (i, vi.name, dims(vi)))
    print('  total=%d' % len(g.input))
    print('--- outputs ---')
    for i, vi in enumerate(g.output):
        if i < 4 or i >= len(g.output) - 2:
            print('  [%2d] %-28s %s' % (i, vi.name, dims(vi)))
    print('  total=%d' % len(g.output))

    hist = collections.Counter(n.op_type for n in g.node)
    print('--- op histogram (%d kinds) ---' % len(hist))
    for k, v in hist.most_common():
        print('  %-26s %d' % (k, v))

    print('--- rope / attention node attrs ---')
    for n in g.node:
        if any(k in n.op_type for k in ('Rotary', 'Attention', 'GQA')) or 'rot' in n.name.lower():
            attrs = []
            for a in n.attribute:
                if a.type == a.INT:
                    attrs.append('%s=%d' % (a.name, a.i))
                elif a.type == a.STRING:
                    attrs.append('%s=%r' % (a.name, a.s.decode('utf-8', 'replace')))
                elif a.type == a.FLOAT:
                    attrs.append('%s=%s' % (a.name, a.f))
                elif a.type == a.INTS:
                    attrs.append('%s=%s' % (a.name, list(a.ints)))
                else:
                    attrs.append('%s(type%d)' % (a.name, a.type))
            print('  %s [%s] in=%d out=%d %s' % (
                n.op_type, n.name, len(n.input), len(n.output), '; '.join(attrs)[:260]))

    # which initializers carry the rope caches
    for ini in g.initializer:
        if 'cos' in ini.name or 'sin' in ini.name or 'cache' in ini.name.lower():
            print('  INIT %-24s %s %s ext=%s' % (
                ini.name, onnx.TensorProto.DataType.Name(ini.data_type), tuple(ini.dims),
                [(d.key, d.value) for d in ini.external_data]))


for arg in sys.argv[1:]:
    census(arg, arg)
