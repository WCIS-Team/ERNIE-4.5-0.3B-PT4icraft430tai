"""Find which producer topology icraft's GQA lowering actually requires, by breaking Qwen3.

Why this file exists. Feeding the three GQA slots of a compiling graph straight from graph inputs
(Q3D, this session) crashes with the same bad-allocation the ERNIE graph crashes with -- Qwen3 gets
to op_id 10 (k_norm/Reshape_2) and dies entering the node, exactly like ERNIE dies after its
k_head_concat. That single control invalidates two earlier conclusions: e_v10 and e_v16, where ERNIE's
slots were replaced by graph inputs, no longer show that provenance is irrelevant. They show that the
probe itself is fatal. So the producer topology is back in the list of suspects, and the only honest
way to rank it is to mutate the graph that is known to parse, one topology at a time, keeping every
shape and value legal.

Qwen3's layer-0 attention as exported (parses -- Q3V1 lowers GQA into ops 12..38):

    qkv MatMul [b,s,4096] -> Split sizes [2048,1024,1024] -> output_0 -> Reshape_1 [b,s,16,128]
      -> SimplifiedLayerNormalization (q_norm) -> Reshape_2 [b,s,2048] -----------------> slot 0
                                                      output_1 -> Reshape_1 [b,s,8,128]
      -> SimplifiedLayerNormalization (k_norm) -> Reshape_2 [b,s,1024] ----------------> slot 1
                                                      output_2 -------------------------------------------------> slot 2

ERNIE's, after split_qkv.py + tile_kv_heads.py (crashes at the node):

    qkv MatMul [b,s,2560] -> Split sizes [2048,256,256] -> output_0 --------------------> slot 0
                              output_1 -> 2-head Split -> 8-way Concat [b,s,1024] -----> slot 1
                              output_2 -> 2-head Split -> 8-way Concat [b,s,1024] -----> slot 2

Three mutations, each isolated:

  direct  slots 0/1/2 = the Split's three outputs. Drops q_norm/k_norm and their Reshapes. Every
          width stays exactly right (2048/1024/1024). Tests whether the Reshape/SLN chain in front of
          Q and K is load-bearing for the pattern -- ERNIE has no per-head norms at all, so if this
          breaks Qwen3 then that absence, not the tiling, is what kills ERNIE, and the fix is to
          synthesise a shape-only Reshape pair (4D -> 3D) in front of Q and K.
  tile    keeps slot 0 as Qwen3's q_norm Reshape_2 but routes K and V through an identity head
          Split + Concat ([128]x8 out and back in order, so the tensor is bit-identical). Tests
          whether a Concat may feed a GQA slot at all -- ERNIE's K/V are Concat outputs.
  ernie   direct for Q plus tile for K/V: the exact ERNIE node topology, on a graph whose geometry
          already compiles. If only this breaks, it is the combination that matters.

Semantics: `direct` changes what the model computes (it drops the real q_norm/k_norm weights), but
these runs are judged only on whether the parse lowers the node, never on numbers. All three keep the
graph checker-clean by asserting no dangling operand and no duplicate producer before saving.

Data: hardlinked from q3i_v1, so each variant costs only its model.onnx.
"""
import argparse
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile')
Q3 = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'qwen3', 'compile', 'fmodel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='q3i_v1')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--mode', required=True,
                    choices=['direct', 'tile', 'ernie', 'normless', 'flat'])
    ap.add_argument('--layer', type=int, default=-1, help='-1 = every layer')
    args = ap.parse_args()

    src_dir = os.path.join(Q3, args.src)
    dst_dir = os.path.join(ROOT, 'fmodel', args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n

    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    picked = [n for i, n in enumerate(gqas) if args.layer < 0 or i == args.layer]
    print('mode=%s layers=%d of %d' % (args.mode, len(picked), len(gqas)))

    # the shared [128]*8 Split sizes constant, exporter-style (a Constant node, not an initializer)
    knh = [a.i for a in gqas[0].attribute if a.name == 'kv_num_heads'][0]
    hd = 128
    sizes_name = '/model/constants/INT64/[%s]' % ', '.join([str(hd)] * knh)
    sizes_const = None
    if args.mode in ('tile', 'ernie'):
        arr = onnx.numpy_helper.from_array(np.array([hd] * knh, dtype=np.int64),
                                           name=sizes_name)
        sizes_const = onnx.helper.make_node(
            'Constant', [], [sizes_name],
            name='/model/constant_nodes/INT64/[%s]' % ', '.join([str(hd)] * knh), value=arr)
        for i in list(g.initializer):
            if i.name == sizes_name:
                g.initializer.remove(i)
        g.node.insert(0, sizes_const)
        g.value_info.append(onnx.helper.make_tensor_value_info(
            sizes_name, onnx.TensorProto.INT64, [knh]))

    changed = 0
    for node in picked:
        idx = gqas.index(node)
        pre = node.name[:node.name.rindex('/')]
        want_direct = args.mode in ('direct', 'ernie')
        want_tile = args.mode in ('tile', 'ernie')
        qkv_split = None
        for s in (0, 1, 2):
            prod = by_out.get(node.input[s])
            if prod is not None and prod.op_type == 'Split' and len(prod.output) == 3:
                qkv_split = prod
        if args.mode == 'normless':
            # keep the head round trip, drop only the q_norm/k_norm SLN in the middle:
            # Split[b,s,W] -> Reshape_1[b,s*heads,128] -> Reshape_2[b,s,W], a pure identity.
            for slot in (0, 1):
                r2 = by_out.get(node.input[slot])
                if r2 is None or r2.op_type != 'Reshape':
                    raise SystemExit('layer %d slot %d: expected Reshape_2, got %s'
                                     % (idx, slot, r2 and r2.op_type))
                sln = by_out.get(r2.input[0])
                if sln is None or sln.op_type != 'SimplifiedLayerNormalization':
                    raise SystemExit('layer %d slot %d: expected an SLN under Reshape_2, got %s'
                                     % (idx, slot, sln and sln.op_type))
                r2.input[0] = sln.input[0]
                g.node.remove(sln)
            changed += 1
            continue
        if args.mode == 'flat':
            # one identity Reshape per slot, 3D -> 3D, straight off the Split: this is e_v12's
            # topology put onto a graph that compiles.
            shape_for = {}
            for slot in (0, 1):
                shape_for[slot] = by_out[node.input[slot]].input[1]
            shape_for[2] = shape_for[1]
            for slot in (0, 1, 2):
                src = qkv_split.output[slot]
                width = int(shape_for[slot].split(',')[-1].strip(' ]'))
                nm = '%s/flat%d' % (pre, slot)
                rs = onnx.helper.make_node('Reshape', [src, shape_for[slot]],
                                           [nm + '/output_0'], name=nm)
                g.node.insert(list(g.node).index(qkv_split) + 1, rs)
                g.value_info.append(onnx.helper.make_tensor_value_info(
                    rs.output[0], onnx.TensorProto.FLOAT16,
                    ['batch_size', 'sequence_length', width]))
                node.input[slot] = rs.output[0]
                by_out[rs.output[0]] = rs
            changed += 1
            continue
        if want_direct:
            if qkv_split is None:
                raise SystemExit('layer %d: no 3-way Split ancestor to take slots from' % idx)
            node.input[0] = qkv_split.output[0]
            if args.mode == 'direct':
                node.input[1] = qkv_split.output[1]
                node.input[2] = qkv_split.output[2]
                changed += 1
                continue
        if want_tile:
            if args.mode == 'ernie':
                node.input[1] = qkv_split.output[1]
                node.input[2] = qkv_split.output[2]
            for slot in (1, 2):
                wide = node.input[slot]
                per = ['%s/tile%d_HeadSplit/output_%d' % (pre, slot, j) for j in range(knh)]
                hs = onnx.helper.make_node('Split', [wide, sizes_name], per,
                                           name='%s/tile%d_head_split' % (pre, slot), axis=-1)
                g.node.insert(list(g.node).index(by_out[wide]) + 1, hs)
                cc = onnx.helper.make_node('Concat', per, ['%s/tile%d_tiled' % (pre, slot)],
                                           name='%s/tile%d_head_concat' % (pre, slot), axis=-1)
                g.node.insert(list(g.node).index(hs) + 1, cc)
                for p in per:
                    g.value_info.append(onnx.helper.make_tensor_value_info(
                        p, onnx.TensorProto.FLOAT16,
                        ['batch_size', 'sequence_length', hd]))
                g.value_info.append(onnx.helper.make_tensor_value_info(
                    cc.output[0], onnx.TensorProto.FLOAT16,
                    ['batch_size', 'sequence_length', knh * hd]))
                node.input[slot] = cc.output[0]
                by_out[hs.output[0]] = hs
                by_out[cc.output[0]] = cc
            changed += 1

    # integrity: no dangling operand, no duplicate producer, no stale value_info
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
    print('layers touched: %d; nodes %d splits %d concats %d'
          % (changed, len(chk.graph.node),
             sum(1 for n in chk.graph.node if n.op_type == 'Split'),
             sum(1 for n in chk.graph.node if n.op_type == 'Concat')))
    print('layer0 GQA slots: %s' % [i[-32:] if i else '' for i in cg.input[:3]])
    for v in chk.graph.value_info:
        if v.name == cg.input[1]:
            print('slot1 dims', [str(d.dim_value or d.dim_param) for d in
                                  v.type.tensor_type.shape.dim])
    print('DONE_MUTATE %s' % args.mode)


if __name__ == '__main__':
    main()
