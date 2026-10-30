"""Rewrite inline-initializer Split sizes operands into the exporter's Constant-node form.

The qkv unpacking (split_qkv.py) and the KV tiling (tile_kv_heads.py) attached their Split `sizes`
operands as g.initializer entries. icraft lowers those Splits fine -- they appear as op_id 4..8 in
the parse log -- but the graph dies with an access violation / bad allocation the moment it reaches
the GroupQueryAttention that consumes them.

q3i_v1, which parses, never does that. Its ONNX Runtime exporter emits exactly 6 Constant nodes
(named /model/constant_nodes/INT64/[...], output /model/constants/INT64/[...], each immediately
before its first consumer) plus a value_info entry for each, and the Split references that output
name; the name is absent from g.initializer. So the sizes operand is a *node-produced* tensor.

This script converts every such operand to that shape, byte-for-byte in style: Constant node +
value_info, initializer removed, same tensor name so nothing downstream has to be rewired.
"""
import argparse
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v7')
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    inits_all = {}
    for i in g.initializer:
        inits_all.setdefault(i.name, []).append(i)
    inits = {k: v[0] for k, v in inits_all.items()}
    dups = {k: len(v) for k, v in inits_all.items() if len(v) > 1}
    print('duplicate initializer names in %s: %d %s' % (args.src, len(dups), list(dups.items())[:3]))

    # which initializer-backed operands are consumed by a node that needs them as a value
    consumers = {}
    for idx, n in enumerate(g.node):
        for slot, inp in enumerate(list(n.input)):
            if inp in inits and n.op_type in ('Split', 'Gather', 'Slice', 'Reshape', 'Expand', 'Tile'):
                if slot == 0 and n.op_type == 'Split':
                    continue
                consumers.setdefault(inp, []).append((idx, n.op_type, n.name, slot))
    targets = {k: v for k, v in consumers.items()
               if k.startswith('/model/constants/')}
    print('operand initializers to constantize: %d' % len(targets))
    if not targets:
        raise SystemExit('nothing matched -- check the name convention before assuming this is the bug')

    insert_at = []
    for name, users in targets.items():
        first = min(u[0] for u in users)
        arr = onnx.numpy_helper.to_array(inits[name])
        const = onnx.helper.make_node(
            'Constant', [], [name],
            name='/model/constant_nodes/%s/%s' % (
                onnx.TensorProto.DataType.Name(inits[name].data_type),
                '[%s]' % ', '.join(str(x) for x in arr.tolist()) if arr.ndim else arr.item()),
            value=onnx.numpy_helper.from_array(arr, name=name + '/value'))
        if len(users) > 1:
            print('  %s (%s %s) consumed by %d nodes, first index %d (%s)'
                  % (name, arr.dtype, arr.shape, len(users), first, users[0][1]))
        insert_at.append((first, const))
        g.value_info.append(onnx.helper.make_tensor_value_info(
            name, inits[name].data_type, list(arr.shape)))

    for pos, const in sorted(insert_at, reverse=True):
        g.node.insert(pos, const)
    removed = 0
    for name in targets:
        for i in inits_all[name]:
            g.initializer.remove(i)
            removed += 1
    print('initializer entries removed: %d for %d names' % (removed, len(targets)))

    out_path = os.path.join(dst_dir, 'model.onnx')
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
    cg = chk.graph
    print('after: nodes %d inits %d constants %d value_info %d'
          % (len(cg.node), len(cg.initializer),
             sum(1 for n in cg.node if n.op_type == 'Constant'), len(cg.value_info)))
    sp = [n for n in cg.node if n.op_type == 'Split'][0]
    ci = list(cg.node).index(sp)
    print('first Split at index %d inputs %s' % (ci, list(sp.input)))
    print('  its sizes operand is an initializer? %s'
          % any(i.name == sp.input[1] for i in cg.initializer))
    for n in cg.node[max(0, ci - 3):ci]:
        if n.op_type == 'Constant':
            print('  preceding Constant: %s -> %s' % (n.name, n.output[0]))
    seen = {}
    for n in cg.node:
        for o in n.output:
            if o and o in seen:
                raise SystemExit('duplicate tensor producer: %r from %s and %s'
                                 % (o, seen[o], n.name))
            seen[o] = n.name
    print('unique tensor producers: ok (%d named outputs)' % len(seen))
    print('onnx', os.path.getsize(out_path), 'bytes')
    print('DONE_CONSTANTIZE')


if __name__ == '__main__':
    main()
