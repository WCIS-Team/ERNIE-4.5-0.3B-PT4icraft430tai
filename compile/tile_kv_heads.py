"""Widen ERNIE's K/V to the head geometry the parser is known to lower.

Where this comes from. With the qkv unpacked (split_qkv.py) the first GroupQueryAttention no longer
access-violates; it throws std::bad_alloc while icraft lowers it. Every candidate scalar was
eliminated one at a time, each time re-running the parse:

  rope cache rows 131072 -> 40960        still bad_alloc   (shrink_rope.py)
  rotary_interleaved 1 -> 0              still bad_alloc   (patch_attr.py)
  value_info declared on the new Splits  still bad_alloc   (split_qkv.py)
  toml max_prompt_length 32->8, max_tokens_length 512->64  still bad_alloc

so the size does not come from the cache geometry or from the toml's sequence lengths. What still
differs from the two graphs that compile is the KV head geometry itself: ERNIE is num_heads 16 /
kv_num_heads 2 / head_dim 128, so K and V are 256 wide while hidden_size is 1024. The vendor
Qwen2-0.5B (kv 2) has K width 128 == kv*head_dim with head_dim == hidden/num_heads = 64; Qwen3-0.6B
(kv 8) has K width 1024 == hidden_size. ERNIE is the only one whose K width is neither.

Fix: replicate each KV head 4 times so the node reads as kv_num_heads=8, K/V 1024 wide -- the exact
Qwen3 shape family -- using only Split and Concat, both already proven in this pipeline (no ONNX
Tile, which no graph here has ever exercised).

The replication is semantically exact. icraft maps Q head h to KV head h*num_kv/num_q. With the true
kv=2 that is h//8, so Q0..Q7 share original head 0. After tiling to 8 new heads ordered
[0,0,0,0,1,1,1,1], Q head h -> new head h//2 -> original (h//2)//4, which is h//8 for every h in
0..15. Verified arithmetically below, and asserted in code.

Cost, and it is real: the KV cache grows 4x (18 layers x 2 x 8 x 128 fp16 = 589,824 B per position,
against 147,456 B for the native geometry), so the driver has to size past/present for 8 heads.
"""
import argparse
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def set_attr(node, name, val):
    for a in node.attribute:
        if a.name == name:
            old = a.i
            a.i = int(val)
            return old
    raise SystemExit('attribute %s missing on %s' % (name, node.name))


def vi_dims(g, name):
    for v in g.value_info:
        if v.name == name:
            return [d.dim_value if d.HasField('dim_value') else d.dim_param
                    for d in v.type.tensor_type.shape.dim]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v4')
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n

    sizes_init = None
    new_knh = None
    touched_cache = 0
    for gqa in [n for n in g.node if n.op_type == 'GroupQueryAttention']:
        nh = [a.i for a in gqa.attribute if a.name == 'num_heads'][0]
        knh = [a.i for a in gqa.attribute if a.name == 'kv_num_heads'][0]
        sp = by_out[gqa.input[0]]
        assert sp.op_type == 'Split', 'expected the split_qkv.py Split, got %s' % sp.op_type
        prod = by_out[sp.input[0]]
        hidden = vi_dims(g, prod.input[0])[-1]
        kw = vi_dims(g, gqa.input[1])[-1]
        vw = vi_dims(g, gqa.input[2])[-1]
        if kw % knh or vw % knh:
            raise SystemExit('K/V width not divisible by kv_num_heads: %s %s %s' % (kw, vw, knh))
        hd = kw // knh
        if vw // knh != hd:
            raise SystemExit('K and V head dims differ')
        if hidden % hd:
            raise SystemExit('hidden %d not a multiple of head_dim %d' % (hidden, hd))
        target = hidden // hd
        rep = target // knh
        if target * hd != hidden or rep * knh != target:
            raise SystemExit('arithmetic: target %d rep %d' % (target, rep))
        if new_knh is None:
            new_knh = target
            print('hidden=%s kv_width=%s -> head_dim=%d; kv_num_heads %d -> %d (each head x%d);'
                  ' K/V width %d -> %d' % (hidden, kw, hd, knh, target, rep, kw, target * hd))
            name = '/model/constants/INT64/[%s]' % ', '.join([str(hd)] * knh)
            sizes_init = onnx.numpy_helper.from_array(
                np.array([hd] * knh, dtype=np.int64), name=name)
            g.initializer.append(sizes_init)
        if target != new_knh:
            raise SystemExit('layer geometry differs from layer 0')

        pre = prod.name[:prod.name.rindex('/')]
        for slot, letter in ((1, 'k'), (2, 'v')):
            wide = gqa.input[slot]
            per = ['%s/%s_HeadSplit/output_%d' % (pre, letter, j) for j in range(knh)]
            hs = onnx.helper.make_node('Split', [wide, sizes_init.name], per,
                                       name='%s/%s_head_split' % (pre, letter), axis=-1)
            g.node.insert(list(g.node).index(by_out[wide]) + 1, hs)
            flat = []
            for j in range(knh):
                flat += [per[j]] * rep
            out = '%s/%s_replicated' % (pre, letter)
            cc = onnx.helper.make_node('Concat', flat, [out],
                                       name='%s/%s_head_concat' % (pre, letter), axis=-1)
            g.node.insert(list(g.node).index(hs) + 1, cc)
            g.value_info.extend(
                onnx.helper.make_tensor_value_info(p, onnx.TensorProto.FLOAT16,
                                                   ['batch_size', 'sequence_length', hd])
                for p in per)
            g.value_info.append(onnx.helper.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT16, ['batch_size', 'sequence_length', target * hd]))
            gqa.input[slot] = out
        old = set_attr(gqa, 'kv_num_heads', target)
        if old != knh:
            raise SystemExit('kv_num_heads was %s, expected %s' % (old, knh))

    # and the cache contract: axis 1 of every past/present tensor
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if not (v.name.startswith('past_key_values.') or v.name.startswith('present.')):
            continue
        dims = v.type.tensor_type.shape.dim
        if dims[1].dim_value != 2:
            raise SystemExit('%s axis1 is %s, expected 2' % (v.name, dims[1].dim_value))
        dims[1].dim_value = new_knh
        touched_cache += 1
    print('cache tensors rewritten: %d (36 inputs + 36 outputs + 0 value_info expected)'
          % touched_cache)

    out_path = os.path.join(dst_dir, 'model.onnx')
    # two nodes must never produce the same tensor name; the first version of this script let the k
    # and v head-splits emit identical names, which is illegal and makes the GQA's inputs ambiguous.
    seen = {}
    for n in g.node:
        for o in n.output:
            if o and o in seen:
                raise SystemExit('duplicate tensor producer: %r from %s and %s'
                                 % (o, seen[o], n.name))
            seen[o] = n.name
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
    cg = [n for n in chk.graph.node if n.op_type == 'GroupQueryAttention'][0]
    print('first GQA kv_num_heads=%d num_heads=%d' % (
        [a.i for a in cg.attribute if a.name == 'kv_num_heads'][0],
        [a.i for a in cg.attribute if a.name == 'num_heads'][0]))
    print('slots:', cg.input[0].split('/')[-1], cg.input[1], cg.input[2])
    for v in chk.graph.input:
        if v.name == 'past_key_values.0.key':
            print('past_key_values.0.key ->', [d.dim_value if d.HasField('dim_value') else d.dim_param
                                               for d in v.type.tensor_type.shape.dim])
    print('nodes %d inits %d splits %d concats %d value_info %d'
          % (len(chk.graph.node), len(chk.graph.initializer),
             sum(1 for n in chk.graph.node if n.op_type == 'Split'),
             sum(1 for n in chk.graph.node if n.op_type == 'Concat'),
             len(chk.graph.value_info)))
    print('onnx', os.path.getsize(out_path), 'bytes; data', os.path.getsize(link), 'bytes')
    print('DONE_TILEKV')


if __name__ == '__main__':
    main()
