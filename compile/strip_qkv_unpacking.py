"""Strip the packed-qkv unpacking chain so nothing in the graph looks like a Split-fed GQA.

Where this comes from. drop_attention.py (e_v15) removed the 18 GroupQueryAttention nodes and the
whole ERNIE graph then parsed: 293 lowered ops, ending at `Pattern : Output ; ftmp_name:
network_output`, with jr_path/E15 written. So the blocker is exactly that one node, and everything
else in ERNIE -- embedding, both norm flavours, the SwiGLU mlp, the mask subgraph, lm_head at vocab
103424 over 19 layers -- lowers without complaint.

Since then the node's own contract has been measured against the compiling Qwen3 graph and found
identical in every observable respect: slots 0/1/2 FLOAT16 [b,s,2048]/[b,s,1024]/[b,s,1024], past and
value [b,8,past_sequence_length,128], mask slot 5 INT32 [batch_size] and slot 6 INT32 scalar from a
byte-for-byte identical attn_mask_reformat subgraph, cos/sin as FLOAT16 initializers of [40960,64]
holding a proper rope table (row0 all ones, no NaN/Inf, same value class as Qwen3's), and the
attribute vector num_heads 16 / kv_num_heads 8 / scale 0.0883883 / local_window_size -1 / softcap 0
/ rotary_interleaved 0 -- e_v13 even carried Qwen3's do_rotary=1, so every scalar has already been
put at the value the working graph uses. Feeding Q/K/V straight from graph inputs (e_v10) did not
help either.

What still differs from Qwen3 is the *shape of the unpacking*: Qwen3's qkv MatMul is 4096 wide and its
3-way Split has sizes [2048,1024,1024] == [q, kv*head_dim, kv*head_dim], with Split/output_2 going
straight into slot 2. ERNIE's qkv MatMul is 2560 wide, Split sizes [2048,256,256], and the 1024-wide
K/V are then rebuilt by 36 extra Splits/Concats. If icraft's GQA handling matches or inspects that
ancestor Split -- to fuse qkv projection with head splitting, or to derive geometry from its sizes --
then a Split whose 256-wide outputs contradict kv_num_heads 8 x head_dim 128 = 1024 is exactly the
kind of inconsistency that can produce a garbage allocation size, and bad_alloc is what we see.

This probe removes that possibility: e_v16 = e_v10 (all three slots already graph inputs) with every
Split and Concat node deleted, so the graph contains no unpacking structure at all while the GQA nodes
keep their graph-input feeds. A hardlink keeps the 967 MB of weights free.

Reading it. Parses -> the unpacking chain is implicated, and the fix is the honest one: replicate the
qkv projection's K/V weight columns 4x so the MatMul is 4096 wide and a single Split of sizes
[2048,1024,1024] feeds slots 0/1/2 -- Qwen3's topology exactly, with no extra nodes. Crashes -> the
node itself cannot be lowered in ERNIE's context for a reason no contract-level edit reaches, and the
only way forward is to hand-decompose the attention into the primitives icraft emits for GQA
(Expand/Cast/RopeDynamic/Reshape/Transpose/Slice/Neg/Concat/Multiply/Add/Matmul/Softmax plus the cache
Concat), which the E15 result says the rest of the pipeline can carry.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v10')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--ops', default='Split,Concat')
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, 'fmodel', args.src)
    dst_dir = os.path.join(ROOT, 'fmodel', args.dst)
    os.makedirs(dst_dir, exist_ok=True)
    drop_ops = {o.strip() for o in args.ops.split(',') if o.strip()}

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph

    removed = [n for n in g.node if n.op_type in drop_ops]
    dead = {o for n in removed for o in n.output}
    still = [(n.name, i) for n in g.node if n.op_type not in drop_ops for i in n.input if i in dead]
    if still:
        raise SystemExit('%d consumers of %s outputs remain: %s'
                         % (len(still), sorted(drop_ops), still[:6]))
    for n in removed:
        g.node.remove(n)
    print('%s nodes removed: %d' % ('/'.join(sorted(drop_ops)), len(removed)))

    # value_info for tensors that no longer exist
    gone = 0
    for v in list(g.value_info):
        if v.name in dead:
            g.value_info.remove(v)
            gone += 1
    print('stale value_info dropped: %d' % gone)

    live = {v.name for v in list(g.input) + list(g.value_info)} | \
           {i.name for i in g.initializer} | \
           {o for n in g.node for o in n.output}
    bad = [(n.name, i) for n in g.node for i in n.input if i and i not in live]
    if bad:
        raise SystemExit('dangling operands: %s' % bad[:6])

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
    print('nodes %d  GQA %d  Split %d  Concat %d'
          % (len(chk.graph.node),
             sum(1 for n in chk.graph.node if n.op_type == 'GroupQueryAttention'),
             sum(1 for n in chk.graph.node if n.op_type == 'Split'),
             sum(1 for n in chk.graph.node if n.op_type == 'Concat')))
    print('layer0 GQA slots:', [i[-24:] if i else '' for i in cg.input[:9]])
    print('op types present: %s' % sorted({n.op_type for n in chk.graph.node}))
    print('DONE_STRIP')


if __name__ == '__main__':
    main()
