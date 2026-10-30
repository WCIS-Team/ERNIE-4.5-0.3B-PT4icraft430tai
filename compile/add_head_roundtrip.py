"""Give ERNIE's GQA the producer topology icraft's lowering turned out to require.

What the controls established (all run on q3i_v1, a graph that parses, mutating layer 0 only so each
result is a single variable; readings are in the Q3* logs under .icraft/logs):

    baseline            slot0/1 <- Reshape_2 <- SLN <- Reshape_1 <- Split, slot2 <- Split  parses
    Q3N  normless       slot0/1 <- Reshape_2 <- Reshape_1 <- Split (SLN deleted), slot2 <- Split
                        PARSES  rc=0, 76 s, 1770 lowered ops, 28 GroupQueryAttention patterns,
                        jr_path/Q3N/ written
    Q3D0 direct         all three slots taken straight off the 3-way Split   -> bad allocation
    Q3F  flat           one identity Reshape (3D -> 3D) per slot             -> bad allocation
    Q3T0 tile           slot0 kept, slots 1/2 fed by identity head Split+Concat -> raw access
                        violation, rc 3221225477 (uncaught, no bad_alloc message)
    Q3E0 ernie          slot0 off the Split, slots 1/2 off Concats           -> bad allocation
    Q3D  decoupled      all three slots replaced by graph inputs             -> bad allocation

So the node needs Q and K to arrive through a two-step head round trip, [b,s,W] -> [b,s*heads,128] ->
[b,s,W]: one Reshape is not enough (Q3F), a Concat is not enough and in fact fails harder (Q3T0), and
the per-head RMSNorm that normally sits in the middle is not what supplies the requirement (Q3N).
This also retires tile_kv_heads.py's approach: a Concat cannot feed a GQA slot at all, so widening K
and V by Split+Concat was never going to compile -- and it is no longer needed, because ERNIE can
keep its native kv_num_heads 2 and its native 147,456 B per position cache.

What this script does to ERNIE's e_v4 (qkv already unpacked by split_qkv.py, geometry untouched):
for every layer, on slots 0 (Q, W=2048) and 1 (K, W=256), insert

    Reshape_1  shape operand [0, -1, 128]   value_info [batch_size,
                 sequence_length * num_attention_heads | sequence_length * num_key_value_heads, 128]
    Reshape_2  shape operand [0, -1, W]     value_info [batch_size, sequence_length, W]

Both shape operands are Constant nodes named and valued exactly the way the onnxruntime-genai exporter
names them, and the node names reuse the exporter's q_norm/k_norm paths (minus the SLN) because that
is the spelling the proven-parsing graph carries. The round trip is a bijection on the same bytes, so
the model computes exactly what it did before. Slot 2 (V) keeps coming from Split/output_2, which is
how Qwen3 feeds V in the graph that compiles. Nothing else in the graph is touched: kv_num_heads stays
2, past/present stay [b,2,past,128], the rope caches stay the exported 131072 x 64 tables, and
do_rotary / rotary_interleaved stay as exported -- those scalars are re-tested separately if this one
still fails.
"""
import argparse
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile')
# exporter symbol names for the merged (sequence x heads) axis, as the working graph declares them
HEAD_SYMBOL = {0: 'num_attention_heads', 1: 'num_key_value_heads'}


def shape_const(g, widths):
    """Return the tensor name of an exporter-style Constant shape operand, creating it once."""
    name = '/model/constants/INT64/[%s]' % ', '.join(str(w) for w in widths)
    if not any(n.name == '/model/constant_nodes/INT64/[%s]'
               % ', '.join(str(w) for w in widths) for n in g.node):
        arr = onnx.numpy_helper.from_array(np.array(widths, dtype=np.int64), name=name + '/value')
        node = onnx.helper.make_node('Constant', [], [name],
                                     name='/model/constant_nodes/INT64/[%s]'
                                          % ', '.join(str(w) for w in widths), value=arr)
        g.node.insert(0, node)
        g.value_info.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT64,
                                                               [len(widths)]))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v4')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--slots', default='0,1')
    ap.add_argument('--head-dim', type=int, default=128)
    ap.add_argument('--interleaved', default=None, choices=['0', '1'],
                    help='force rotary_interleaved on every GQA (probe only, changes semantics)')
    ap.add_argument('--do-rotary', default=None, choices=['0', '1'])
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, 'fmodel', args.src)
    dst_dir = os.path.join(ROOT, 'fmodel', args.dst)
    os.makedirs(dst_dir, exist_ok=True)
    slots = [int(s) for s in args.slots.split(',')]

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n
    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    def dims_of(name):
        v = vi.get(name)
        if v is None:
            raise SystemExit('no declared shape for %s' % name)
        return [d.dim_value if d.HasField('dim_value') else d.dim_param
                for d in v.type.tensor_type.shape.dim]

    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    if not gqas:
        raise SystemExit('no GroupQueryAttention in %s' % args.src)
    nh = [a.i for a in gqas[0].attribute if a.name == 'num_heads'][0]
    knh = [a.i for a in gqas[0].attribute if a.name == 'kv_num_heads'][0]

    added = 0
    for node in gqas:
        idx = gqas.index(node)
        for slot in slots:
            width = dims_of(node.input[slot])[-1]
            heads = width // args.head_dim
            if heads * args.head_dim != width:
                raise SystemExit('layer %d slot %d: width %s not a multiple of head_dim %d'
                                 % (idx, slot, width, args.head_dim))
            if heads != (nh if slot == 0 else knh):
                raise SystemExit('layer %d slot %d: %d heads, attributes say %d'
                                 % (idx, slot, heads, nh if slot == 0 else knh))
            sym = '%s * %s' % (dims_of(node.input[slot])[1], HEAD_SYMBOL[slot])
            src = node.input[slot]
            prod = by_out[src]
            pre = node.name[:node.name.rindex('/')]
            norm = 'q_norm' if slot == 0 else 'k_norm'
            r1 = onnx.helper.make_node(
                'Reshape', [src, shape_const(g, [0, -1, args.head_dim])],
                ['%s/%s/Reshape_1/output_0' % (pre, norm)], name='%s/%s/Reshape_1' % (pre, norm))
            r2 = onnx.helper.make_node(
                'Reshape', [r1.output[0], shape_const(g, [0, -1, width])],
                ['%s/%s/Reshape_2/output_0' % (pre, norm)], name='%s/%s/Reshape_2' % (pre, norm))
            pos = list(g.node).index(prod) + 1
            g.node.insert(pos, r1)
            g.node.insert(pos + 1, r2)
            g.value_info.append(onnx.helper.make_tensor_value_info(
                r1.output[0], onnx.TensorProto.FLOAT16,
                [dims_of(node.input[slot])[0], sym, str(args.head_dim)]))
            g.value_info.append(onnx.helper.make_tensor_value_info(
                r2.output[0], onnx.TensorProto.FLOAT16,
                [str(x) for x in dims_of(node.input[slot])]))
            node.input[slot] = r2.output[0]
            by_out[r1.output[0]] = r1
            by_out[r2.output[0]] = r2
            vi[r1.output[0]] = g.value_info[-2]
            vi[r2.output[0]] = g.value_info[-1]
            added += 1
        if args.interleaved is not None:
            for a in node.attribute:
                if a.name == 'rotary_interleaved':
                    a.i = int(args.interleaved)
        if args.do_rotary is not None:
            for a in node.attribute:
                if a.name == 'do_rotary':
                    a.i = int(args.do_rotary)
    print('round trips inserted: %d (slots %s, head_dim %d)' % (added, slots, args.head_dim))

    live = ({v.name for v in list(g.input) + list(g.value_info)}
            | {i.name for i in g.initializer}
            | {o for n in g.node for o in n.output})
    bad = [(n.name, i) for n in g.node for i in n.input if i and i not in live]
    if bad:
        raise SystemExit('dangling operands: %s' % bad[:6])
    seen = {}
    for n in g.node:
        for o in n.output:
            if o and o in seen:
                raise SystemExit('duplicate producer %r: %s and %s' % (o, seen[o], n.name))
            seen[o] = n.name

    onnx.save(m, os.path.join(dst_dir, 'model.onnx'))
    link = os.path.join(dst_dir, 'model.onnx.data')
    if not os.path.exists(link):
        os.link(os.path.join(src_dir, 'model.onnx.data'), link)
        print('hardlinked data', os.path.getsize(link), 'bytes')
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst_dir, extra), 'w', encoding='utf-8',
                 newline='\n').write(open(s, encoding='utf-8').read())

    chk = onnx.load(os.path.join(dst_dir, 'model.onnx'), load_external_data=False)
    cg = [n for n in chk.graph.node if n.op_type == 'GroupQueryAttention'][0]
    print('nodes %d  reshape %d  split %d  concat %d  value_info %d'
          % (len(chk.graph.node), sum(1 for n in chk.graph.node if n.op_type == 'Reshape'),
             sum(1 for n in chk.graph.node if n.op_type == 'Split'),
             sum(1 for n in chk.graph.node if n.op_type == 'Concat'),
             len(chk.graph.value_info)))
    print('layer0 GQA slots: %s' % [i[-34:] if i else '' for i in cg.input[:3]])
    print('attrs: %s' % {a.name: (a.i if a.i else a.f) for a in cg.attribute})
    for v in chk.graph.input:
        if v.name == 'past_key_values.0.key':
            print('cache contract: %s' % [str(d.dim_value or d.dim_param)
                                          for d in v.type.tensor_type.shape.dim])
    print('DONE_ROUNDTRIP')


if __name__ == '__main__':
    main()
