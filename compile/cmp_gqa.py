"""Side-by-side dump of the FIRST GroupQueryAttention node in two graphs.

The ERNIE parse dies with an access violation at op_id 4 (the first GQA) while the Qwen3 graph
compiled through the identical recipe, so the difference has to be in that node: its attribute
values, its input list (order and who produces each input), or the declared dtype/shape of the
cos/sin caches and the mask. Print all three so the mismatch is visible in one screen.
"""
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork')
GRAPHS = {
    'ERNIE': os.path.join(ROOT, 'ernie', 'compile', 'fmodel', 'e_v3', 'model.onnx'),
    'Qwen3': os.path.join(ROOT, 'qwen3', 'compile', 'fmodel', 'q3i_v1', 'model.onnx'),
}


def dt(t):
    return onnx.TensorProto.DataType.Name(t)


def init_info(g, name):
    for i in g.initializer:
        if i.name == name:
            return 'INIT %s %s' % (dt(i.data_type), list(i.dims))
    return None


def vi_info(g, name):
    for coll in (g.value_info, g.input, g.output):
        for v in coll:
            if v.name == name:
                tt = v.type.tensor_type
                dims = [d.dim_value if d.HasField('dim_value') else d.dim_param
                        for d in tt.shape.dim]
                return 'VINFO %s %s' % (dt(tt.elem_type), dims)
    return None


def describe(g, name):
    if not name:
        return '<EMPTY SLOT>'
    prod = [n for n in g.node if name in list(n.output)]
    if prod:
        return 'FROM %s %s | %s' % (prod[0].op_type, prod[0].name, vi_info(g, name) or 'no value_info')
    return init_info(g, name) or vi_info(g, name) or '???'


def main():
    graphs = dict(GRAPHS)
    for a in sys.argv[1:]:
        graphs[os.path.basename(os.path.dirname(a))] = a
    for label, path in graphs.items():
        if not os.path.exists(path):
            print('%s: MISSING %s' % (label, path))
            continue
        m = onnx.load(path, load_external_data=False)
        g = m.graph
        gqa = [n for n in g.node if n.op_type == 'GroupQueryAttention']
        print('=' * 70)
        if not gqa:
            att = sorted({n.op_type for n in g.node if 'ttention' in n.op_type or 'otary' in n.op_type})
            print('%s: NO GroupQueryAttention; attention-ish op types = %s; total nodes %d'
                  % (label, att, len(g.node)))
            continue
        print('%s: %d GQA nodes, first = %s' % (label, len(gqa), gqa[0].name))
        print('-- attributes --')
        for a in sorted(gqa[0].attribute, key=lambda x: x.name):
            print('  %-24s type=%d value=%s' % (a.name, a.type, onnx.helper.get_attribute_value(a)))
        print('-- inputs (%d) --' % len(gqa[0].input))
        for k, inp in enumerate(gqa[0].input):
            print('  [%d] %-46s %s' % (k, inp, describe(g, inp)))
        print('-- outputs --')
        for o in gqa[0].output:
            print('  %-46s %s' % (o, describe(g, o)))
        print('-- cos/sin cache initializers anywhere --')
        for i in g.initializer:
            if 'cos' in i.name or 'sin' in i.name:
                print('  %-52s %s %s' % (i.name, dt(i.data_type), list(i.dims)))
        print('-- mask/position reformat subgraph --')
        for n in g.node:
            if 'attn_mask_reformat' in n.name or 'position' in n.name:
                outs = ['%s=%s' % (o, vi_info(g, o) or '?') for o in n.output]
                print('  %-26s %-46s %s -> %s' % (n.op_type, n.name, list(n.input), '; '.join(outs)))


if __name__ == '__main__':
    main()
