"""Un-pack ERNIE's fused qkv into the separate Q/K/V inputs the parser demonstrably handles.

Why this is the fix and not a guess. icraft lowers a GroupQueryAttention itself; its own parse logs
show how it slices the projection:

  vendor Qwen2-0.5B (fused qkv, parses):  Slice { 1 32 1152 } -> { 1 32 896 }  -> Reshape { 1 32 14 64 }
                                          Slice { 1 32 1152 } -> { 1 32 128 }
  Qwen3-0.6B (split qkv, parses):         Reshape { 1 32 2048 } -> { 1 32 16 128 }

So in the fused form the parser takes Q width = hidden_size and derives head_dim = hidden/num_heads.
That is only self-consistent when head_dim == hidden/num_heads (Qwen2: 896/14 = 64). ERNIE has
hidden 1024, num_heads 16, head_dim 128 (config.json says head_dim=128, and the qkv weight
{1024 2560} = 16*128 + 2*128 + 2*128), so the parser's derived 64 contradicts the 128 the past/present
cache axes declare, and the access violation lands exactly on that first GQA. Flipping
rotary_interleaved and baking kv_cache_dim were both tested and neither moved the crash.

The split form makes every width explicit, so head_dim comes from the tensor, not from hidden. The
node definition copies the proven Qwen3 exporter form: Split with attribute axis=-1 and an INT64
sizes initializer named /model/constants/INT64/[...] (inline, not external data).

Semantics are unchanged: ERNIE has use_bias=false, so the MatMul output is the raw fused projection
and slicing it on the last axis is exactly what ORT's GQA documents for the key/value-provided form.
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
    ap.add_argument('--src', default='e_v2')
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    by_out = {n.output[0]: n for n in g.node if len(n.output) == 1}

    sizes_init = None
    rewritten = 0
    for node in [n for n in g.node if n.op_type == 'GroupQueryAttention']:
        nh = [a.i for a in node.attribute if a.name == 'num_heads'][0]
        knh = [a.i for a in node.attribute if a.name == 'kv_num_heads'][0]
        assert not node.input[1] and not node.input[2], \
            '%s already has key/value -- expected fused form' % node.name
        prod = by_out.get(node.input[0])
        assert prod is not None and prod.op_type == 'MatMul', \
            '%s fed by %s, expected the qkv MatMul' % (node.name, prod)
        w = inits[prod.input[1]]
        total = w.dims[1]
        if total % (nh + 2 * knh):
            raise SystemExit('qkv width %d not divisible by %d heads' % (total, nh + 2 * knh))
        hd = total // (nh + 2 * knh)
        q, kv = nh * hd, knh * hd
        if q + 2 * kv != total:
            raise SystemExit('arithmetic mismatch %d + 2*%d != %d' % (q, kv, total))

        if sizes_init is None:
            print('head_dim=%d q=%d k=%d v=%d (from qkv width %d, num_heads %d, kv_num_heads %d)'
                  % (hd, q, kv, kv, total, nh, knh))
            # cross-check against the declared cache axis, which bake_kv_dim.py set from config.json
            past = [v for v in g.input if v.name.startswith('past_key_values.0.key')][0]
            axis = past.type.tensor_type.shape.dim[-1]
            if not axis.HasField('dim_value') or axis.dim_value != hd:
                raise SystemExit('past cache last axis %s disagrees with derived head_dim %d'
                                 % (axis.dim_param or axis.dim_value, hd))
            print('cross-check ok: past_key_values.0.key declares head_dim %d' % hd)
            name = '/model/constants/INT64/[%d, %d, %d]' % (q, kv, kv)
            sizes_init = onnx.numpy_helper.from_array(
                np.array([q, kv, kv], dtype=np.int64), name=name)
            g.initializer.append(sizes_init)

        idx = list(g.node).index(prod)
        outs = ['%s/Split/output_%d' % (prod.name[:prod.name.rindex('/')], k) for k in range(3)]
        sp = onnx.helper.make_node('Split', [prod.output[0], sizes_init.name], outs,
                                   name='%s/Split' % prod.name[:prod.name.rindex('/')], axis=-1)
        g.node.insert(idx + 1, sp)
        node.input[0], node.input[1], node.input[2] = outs
        # the exporter declares a value_info for every tensor the GQA reads, so do the same --
        # icraft binds the GQA's head geometry from those declarations
        for o, w in zip(outs, (q, kv, kv)):
            g.value_info.append(onnx.helper.make_tensor_value_info(
                o, onnx.TensorProto.FLOAT16, ['batch_size', 'sequence_length', w]))
        rewritten += 1

    print('GQA nodes rewired: %d (expected 18)' % rewritten)
    if rewritten != 18:
        raise SystemExit('unexpected GQA count')

    out = os.path.join(dst_dir, 'model.onnx')
    onnx.save(m, out)
    link = os.path.join(dst_dir, 'model.onnx.data')
    if not os.path.exists(link):
        try:
            os.link(os.path.join(src_dir, 'model.onnx.data'), link)
            print('hardlinked data')
        except OSError as e:
            print('hardlink failed (%s), copying' % e)
            import shutil
            shutil.copyfile(os.path.join(src_dir, 'model.onnx.data'), link)
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst_dir, extra), 'w', encoding='utf-8', newline='\n').write(
                open(s, encoding='utf-8').read())

    chk = onnx.load(out, load_external_data=False)
    cg = [n for n in chk.graph.node if n.op_type == 'GroupQueryAttention'][0]
    print('first GQA inputs now:', list(cg.input[:3]))
    print('nodes %d (was 209) inits %d (was 131) splits %d value_info %d (was 373)'
          % (len(chk.graph.node), len(chk.graph.initializer),
             sum(1 for n in chk.graph.node if n.op_type == 'Split'), len(chk.graph.value_info)))
    pos = {}
    for k, n in enumerate(chk.graph.node):
        for o in n.output:
            pos[o] = k
    for n in chk.graph.node:
        if n.op_type == 'Split' and n.input[0] in pos and pos[n.input[0]] > list(chk.graph.node).index(n):
            raise SystemExit('Split %s precedes its producer' % n.name)
    print('topological order ok (every Split follows its MatMul)')
    print('onnx', os.path.getsize(out), 'bytes; data', os.path.getsize(link), 'bytes')
    print('DONE_SPLITQKV')


if __name__ == '__main__':
    main()
