"""Put the GQA's Q/K/V behind identity Reshapes, because that is what every graph icraft accepts has.

The graph-context census (cmp_graph_context.py) is now down to one difference that a GQA lowering
would plausibly care about:

    op type        q3i_v1 (parses)   e_v11 (dies entering the GQA)
    Reshape            112                       0
    Concat               0                      36
    SimplifiedLayerNorm 57                       1

q3i_v1's exporter puts a Reshape on the GQA's Q and K slots (they are the outputs of the per-head
q_norm/k_norm chains, reshaped back to [batch, seq, heads*head_dim] with shape constants
[0, -1, 2048] / [0, -1, 1024]) and V straight off the Split. ERNIE has no per-head norms, so its
exporter had nothing to reshape and the GQA is fed bare Split/Concat outputs. head_dim 128 against
hidden_size 1024 with 16 heads means the head layout is exactly the thing the lowering has to know,
so the missing Reshape is the prime suspect rather than a coincidence.

This inserts a no-op Reshape on all three slots of every layer, reusing the exporter's own conventions:
one shared INT64 shape constant per width, emitted as a Constant node (not an initializer) named
/model/constants/INT64/[0, -1, W], plus a value_info on the output. Semantics are unchanged -- the
reshape is identity -- so if the parse clears, this rewrite can stay in the shipped chain.
"""
import argparse
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
SLOT_HEADS = {0: 16, 1: 8, 2: 8}  # informational only; widths are read from the graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v11')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--layers', default='all', help='"all" or a 0-based layer index')
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    vi = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        vi[v.name] = [d.dim_value if d.HasField('dim_value') else d.dim_param
                      for d in v.type.tensor_type.shape.dim]

    # one shared shape constant per width, in the exporter's Constant-node style
    consts = {}

    def shape_const(width):
        name = '/model/constants/INT64/[0, -1, %d]' % width
        if name not in consts:
            arr = np.array([0, -1, width], dtype=np.int64)
            node = onnx.helper.make_node(
                'Constant', [], [name],
                name='/model/constant_nodes/INT64/[0, -1, %d]' % width,
                value=onnx.numpy_helper.from_array(arr, name=name + '/value'))
            g.value_info.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.INT64, [3]))
            consts[name] = node
        return name, consts[name]

    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    picked = gqas if args.layers == 'all' else [gqas[int(args.layers)]]
    print('GQA nodes %d, patching %d' % (len(gqas), len(picked)))

    new_nodes = []
    for n in picked:
        layer = gqas.index(n)
        for slot in (0, 1, 2):
            old = n.input[slot]
            d = vi.get(old)
            if d is None or not isinstance(d[-1], int):
                raise SystemExit('layer %d slot %d %s has no static width (%s)' % (layer, slot, old, d))
            width = d[-1]
            cname, cnode = shape_const(width)
            out = '%s/in%d_reshape' % (n.name, slot)
            rs = onnx.helper.make_node('Reshape', [old, cname], [out],
                                       name='%s/in%d_Reshape' % (n.name, slot))
            g.value_info.append(onnx.helper.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT16, ['batch_size', 'sequence_length', width]))
            n.input[slot] = out
            new_nodes.append((layer, slot, width, rs, cnode))
        print('  layer %d: %s' % (layer,
                                  ['slot%d width%d' % (s, w) for L, s, w, _, _ in new_nodes
                                   if L == layer]))

    # constants first (before any consumer), so the graph stays topologically ordered
    seen = set()
    for _, _, _, _, cnode in new_nodes:
        if cnode.output[0] not in seen:
            g.node.insert(0, cnode)
            seen.add(cnode.output[0])

    # and each Reshape directly before its own consumer
    for layer, slot, width, rs, _ in new_nodes:
        idx = list(g.node).index(gqas[layer])
        g.node.insert(idx, rs)

    out_path = os.path.join(dst_dir, 'model.onnx')
    seen_prod = {}
    for n in g.node:
        for o in n.output:
            if o and o in seen_prod:
                raise SystemExit('duplicate tensor producer %r (%s, %s)' % (o, seen_prod[o], n.name))
            seen_prod[o] = n.name
    onnx.save(m, out_path)
    link = os.path.join(dst_dir, 'model.onnx.data')
    if not os.path.exists(link):
        os.link(os.path.join(src_dir, 'model.onnx.data'), link)
        print('hardlinked data')
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst_dir, extra), 'w', encoding='utf-8', newline='\n').write(
                open(s, encoding='utf-8').read())

    chk = onnx.load(out_path, load_external_data=False)
    print('after: nodes %d constants %d reshapes %d concats %d splits %d value_info %d'
          % (len(chk.graph.node),
             sum(1 for n in chk.graph.node if n.op_type == 'Constant'),
             sum(1 for n in chk.graph.node if n.op_type == 'Reshape'),
             sum(1 for n in chk.graph.node if n.op_type == 'Concat'),
             sum(1 for n in chk.graph.node if n.op_type == 'Split'),
             len(chk.graph.value_info)))
    first = [n for n in chk.graph.node if n.op_type == 'GroupQueryAttention'][0]
    print('first GQA slots: %s' % list(first.input[:3]))
    # topological sanity: every producer must appear before its consumer
    pos = {o: i for i, n in enumerate(chk.graph.node) for o in n.output}
    inputs = {v.name for v in chk.graph.input}
    inits = {i.name for i in chk.graph.initializer}
    dangling = [(i, n.name, a) for i, n in enumerate(chk.graph.node) for a in n.input
                if a and a not in inputs and a not in inits and a not in pos]
    print('dangling operands: %d %s' % (len(dangling), dangling[:3]))
    late = [(i, n.name, a, pos[a]) for i, n in enumerate(chk.graph.node) for a in n.input
            if a in pos and pos[a] > i]
    print('operands produced after their consumer: %d %s' % (len(late), late[:3]))
    print('onnx', os.path.getsize(out_path), 'bytes')
    print('DONE_RESHAPE')


if __name__ == '__main__':
    main()
