"""Canonicalise the layer-0 attention region of two graphs and print every field that differs.

The GQA node itself was already diffed field by field (dump_gqa_signature.py) and e_v25's layer-0
attention now matches the parsing Qwen3 graph there, yet the parse dies at the same op in both. So the
last unexplored surface is everything *around* the node: the Split's attributes, each producer chain,
each tensor's dtype/shape as the graph declares it, and the value_info set. This prints one sorted
line per (node, field) for each graph and diffs them, with names normalised so that layers.N and the
model name do not create false differences.
"""
import os
import re
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def norm(s):
    s = str(s)
    s = re.sub(r'layers\.\d+', 'layers.N', s)
    return s


def shape_of(vi):
    return [str(d.dim_value if d.HasField('dim_value') else d.dim_param)
            for d in vi.type.tensor_type.shape.dim]


def region(graph, hops=14):
    g = graph
    nodes = list(g.node)
    bo = {}
    for n in nodes:
        for o in n.output:
            if o:
                bo[o] = n
    inits = {i.name: i for i in g.initializer}
    vis = {}
    for v in list(g.input) + list(g.output) + list(g.value_info):
        vis[v.name] = v
    gqa = [n for n in nodes if n.op_type == 'GroupQueryAttention'][0]
    keep = set()
    frontier = [gqa]
    depth = 0
    while frontier and depth < hops:
        nxt = []
        for n in frontier:
            keep.add(id(n))
            for ip in n.input:
                p = bo.get(ip)
                if p is not None and id(p) not in keep:
                    nxt.append(p)
                if ip in vis and vis[ip].name in [x.name for x in g.input]:
                    keep.add(('input', ip))
        frontier = [n for n in nxt if id(n) not in keep]
        depth += 1
    # one consumer level: what the GQA's outputs feed
    for n in nodes:
        if any(ip in gqa.output for ip in n.input):
            keep.add(id(n))
    lines = []
    for n in nodes:
        if id(n) not in keep:
            continue
        attrs = []
        for a in sorted(n.attribute, key=lambda x: x.name):
            attrs.append('%s=%s' % (a.name, norm(a.i if a.type == 2 else
                                                 (a.f if a.type == 1 else
                                                  onnx.numpy_helper.to_array(a.t).tolist()
                                                  if a.type == 3 else a.s))))
        ins = []
        for ip in n.input:
            if ip == '':
                ins.append('<empty>')
                continue
            d = t = 'none'
            if ip in inits:
                d = 'INIT'
                t = '%s%s' % (onnx.TensorProto.DataType.Name(inits[ip].data_type),
                              'EXT' if inits[ip].data_location else '')
                shp = str(list(inits[ip].dims))
            elif ip in bo:
                d = bo[ip].op_type
                shp = str(shape_of(vis[ip])) if ip in vis else 'no-shape'
            elif ip in vis:
                d = 'GRAPH_INPUT'
                shp = str(shape_of(vis[ip]))
            else:
                shp = 'DANGLING'
            ins.append('%s<-%s:%s' % (norm(ip), d, shp))
        outs = []
        for o in n.output:
            vi = vis.get(o)
            outs.append('%s:%s%s' % (norm(o),
                                     onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
                                     if vi is not None else 'no-value_info',
                                     shape_of(vi) if vi is not None else ''))
        lines.append('%-14s %-58s attrs[%s] in[%s] out[%s]'
                     % (n.op_type, norm(n.name), ' '.join(attrs), ' | '.join(ins), ' | '.join(outs)))
    return lines


def main():
    a, b = sys.argv[1], sys.argv[2]
    la = region(onnx.load(os.path.join(ROOT, a, 'model.onnx'), load_external_data=False).graph)
    lb = region(onnx.load(os.path.join(ROOT, b, 'model.onnx'), load_external_data=False).graph)
    print('%s region lines %d; %s region lines %d' % (a, len(la), b, len(lb)))
    ka = {l.split(' attrs[')[0]: l for l in la}
    kb = {l.split(' attrs[')[0]: l for l in lb}
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    print('--- only in %s (%d) ---' % (a, len(only_a)))
    for k in only_a:
        print('  ', k[:150])
    print('--- only in %s (%d) ---' % (b, len(only_b)))
    for k in only_b:
        print('  ', k[:150])
    print('--- same node, different fields ---')
    n = 0
    for k in sorted(set(ka) & set(kb)):
        if ka[k] != kb[k]:
            n += 1
            print('  A:', ka[k][:400])
            print('  B:', kb[k][:400])
    print('differing shared nodes: %d' % n)


if __name__ == '__main__':
    main()
