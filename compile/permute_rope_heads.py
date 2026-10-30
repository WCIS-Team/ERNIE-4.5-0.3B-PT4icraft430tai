"""Convert ERNIE's Q/K head_dim axis from RoPE-interleaved to split-half order.

Where this comes from. The exporter writes the graph with rotary_interleaved=1 and the cos/sin caches
in the matching per-pair layout, i.e. rotation pair i acts on dims (2i, 2i+1). Measured against HF on
the identical HF-seeded feed (6-token prefix cached, 7th token fed):

  native export, ORT                 maxabs 1.459   cos 0.99411
  our rewrite, ORT, ri=1             maxabs 1.459   cos 0.99411
  our rewrite, ORT, ri=0             maxabs 4.235   cos 0.96465
  compiled E38 (ri=0), Host backend  maxabs 4.225   cos 0.96465
  compiled E40 (ri=1), Host backend  maxabs 4.225   cos 0.96465   <- bit-identical to E38

E38 and E40 differ only in that attribute, and their parsed IRs are the same 795,009 bytes apart from
two characters of the net name, so iCraft's onnx parser drops rotary_interleaved entirely: its
RopeDynamicNode always pairs (d, d+head_dim/2). The compiled net therefore reproduces ORT-with-ri=0,
not HF, and no attribute value can change that.

Fix in the data instead. Permute the head_dim axis of Q and K so that the pair iCraft forms --
(i, i+64) with angle i -- carries HF's dims (2i, 2i+1). That is a permutation of the output columns of
qkv_proj inside every Q head block and every K head block; it leaves the attention dot products
unchanged because Q and K are permuted identically, leaves V and so o_proj untouched, and needs no new
nodes. q_norm/k_norm would also need their per-dim gains permuted, but this export has none (the
exporter folded them into qkv_proj: the only q_norm/k_norm nodes are Reshapes), which the script
asserts rather than assumes.

Consequence for the cache feed: present.key is the post-rotary K in the node's own dim order, so an
HF-derived K cache has to be permuted along its last axis the same way (V must not). See
permute_seed_k.py.

The rotary_interleaved attribute is set to 0 in the output graph because after the permutation the
data really is split-half; that keeps an ORT control of this variant correct. It does not affect
iCraft, which is the whole point.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def ext(i):
    return {e.key: e.value for e in i.external_data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v40')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--head-dim', type=int, default=128)
    args = ap.parse_args()

    hd = args.head_dim
    if hd % 2:
        raise SystemExit('head_dim must be even')
    half = hd // 2
    # new position d takes old position perm[d]: pairs (i, i+half) must carry old dims (2i, 2i+1)
    perm = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
    if sorted(perm.tolist()) != list(range(hd)):
        raise SystemExit('perm is not a bijection')

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

    data_src = os.path.join(src_dir, 'model.onnx.data')
    data_dst = os.path.join(dst_dir, 'model.onnx.data')
    shutil.copyfile(data_src, data_dst)
    print('copied data %d bytes' % os.path.getsize(data_dst))

    touched = []
    for gqa in [n for n in g.node if n.op_type == 'GroupQueryAttention']:
        layer = gqa.name.split('/')[2]
        num_heads = [a.i for a in gqa.attribute if a.name == 'num_heads'][0]
        knh = [a.i for a in gqa.attribute if a.name == 'kv_num_heads'][0]
        q = gqa.input[0]
        k = gqa.input[1]
        v = gqa.input[2]
        # the qkv Split is what fixes the column ranges of the packed projection; Q and K reach the node
        # through the exporter's reshape pairs, so walk back through Reshapes to get at it
        split = by_out[q]
        while split.op_type == 'Reshape':
            split = by_out[split.input[0]]
        assert split.op_type == 'Split', split.op_type
        sizes_init = inits.get(split.input[1])
        if sizes_init is None:
            # this export puts its shape constants in Constant nodes, not initializers
            prod = by_out[split.input[1]]
            if prod.op_type != 'Constant':
                raise SystemExit('Split sizes %r come from %s' % (split.input[1], prod.op_type))
            val = [a for a in prod.attribute if a.name == 'value'][0]
            sizes = list(val.t.int64_data)
        else:
            sizes = numpy_helper.to_array(sizes_init).tolist()
        w_name = by_out[split.input[0]].input[1]
        w = inits[w_name]
        dims = list(w.dims)
        if dims[1] != sum(sizes):
            raise SystemExit('%s width %s != split sizes %s' % (w_name, dims[1], sizes))
        ed = ext(w)
        if 'offset' not in ed or int(ed['length']) != dims[0] * dims[1] * 2:
            raise SystemExit('%s is not a flat fp16 external tensor: %s' % (w_name, ed))
        # Q block then K block; the V block keeps HF's dim order, so o_proj needs no change
        q_end = num_heads * hd
        k_end = q_end + knh * hd
        if sizes[0] != q_end or sizes[1] != knh * hd:
            raise SystemExit('layer %s: split %s vs heads %d/%d' % (layer, sizes, num_heads, knh))
        with open(data_dst, 'r+b') as f:
            f.seek(int(ed['offset']))
            arr = np.frombuffer(f.read(int(ed['length'])), dtype=np.float16).reshape(dims)
            new = arr.copy()
            for start in range(0, k_end, hd):
                new[:, start:start + hd] = arr[:, start + perm]
            assert (new[:, k_end:] == arr[:, k_end:]).all(), 'V block moved'
            inv = np.argsort(perm)
            for start in range(0, k_end, hd):
                assert np.array_equal(new[:, start:start + hd][:, inv],
                                      arr[:, start:start + hd]), 'not a permutation'
            f.seek(int(ed['offset']))
            f.write(new.tobytes())
        touched.append((layer, w_name, q_end, knh * hd, int(ed['offset'])))

    # no per-dim q/k norm gains anywhere in this export -- if that stops being true this script is wrong
    gains = [i.name for i in g.initializer
             if ('q_norm' in i.name or 'k_norm' in i.name) and list(i.dims) in ([hd], [1024])]
    norm_nodes = [n.name for n in g.node
                  if n.op_type in ('SkipSimplifiedLayerNormalization', 'SimplifiedLayerNormalization')
                  and ('q_norm' in n.name or 'k_norm' in n.name)]
    print('q/k norm gain initializers %d %s; q/k norm nodes %d %s'
          % (len(gains), gains[:3], len(norm_nodes), norm_nodes[:3]))
    if gains or norm_nodes:
        raise SystemExit('q/k norms are per-dim here; their weights would need the same permutation')

    ri = []
    for gqa in [n for n in g.node if n.op_type == 'GroupQueryAttention']:
        for a in gqa.attribute:
            if a.name == 'rotary_interleaved':
                ri.append(a.i)
                a.i = 0
    print('rotary_interleaved was %s on every GQA -> 0 (%d nodes)' % (sorted(set(ri)), len(ri)))

    print('layers rewritten: %d' % len(touched))
    for t in touched[:2] + touched[-1:]:
        print('  %s  %s  Q[0:%d] K[%d:%d]  data offset %d' % (t[0], t[1], t[2], t[2], t[2] + t[3], t[4]))

    onnx.save(m, os.path.join(dst_dir, 'model.onnx'))
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(dst_dir, extra))

    # read back one layer and check, for every Q head block, that new dim half really is old dim 1
    chk = onnx.load(os.path.join(dst_dir, 'model.onnx'), load_external_data=False)
    cw = {i.name: i for i in chk.graph.initializer}['model.layers.0.attn.qkv_proj.MatMul.weight']
    ced = ext(cw)
    with open(data_dst, 'rb') as f:
        f.seek(int(ced['offset']))
        back = np.frombuffer(f.read(int(ced['length'])), dtype=np.float16)
    with open(data_src, 'rb') as f:
        f.seek(int(ced['offset']))
        orig = np.frombuffer(f.read(int(ced['length'])), dtype=np.float16)
    o = orig.reshape(list(cw.dims))
    b = back.reshape(list(cw.dims))
    old_cols = np.concatenate([np.arange(s, s + hd)[perm] for s in range(0, q_end, hd)])
    new_cols = np.concatenate([np.arange(s, s + hd) for s in range(0, q_end, hd)])
    print('check: all %d Q columns round-trip -> %s'
          % (q_end, bool(np.array_equal(b[:, new_cols], o[:, old_cols]))))
    print('check: new[:, 64] == old[:, 1] for every Q head -> %s'
          % bool(all(np.array_equal(b[:, s + half], o[:, s + 1]) for s in range(0, q_end, hd))))
    print('model.onnx %d bytes, data %d bytes'
          % (os.path.getsize(os.path.join(dst_dir, 'model.onnx')), os.path.getsize(data_dst)))
    print('DONE_PERMROPE')


if __name__ == '__main__':
    main()
