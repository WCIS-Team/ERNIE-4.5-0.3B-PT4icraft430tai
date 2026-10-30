"""Widen ERNIE's K and V natively, by replicating qkv projection columns, so no Concat feeds the GQA.

Why columns and not nodes. The Qwen3 controls (mutate_qwen3_topo.py) fixed the producer *topology* --
Q and K must arrive through a two-step head Reshape round trip, and a Concat is not accepted at all --
but ERNIE still dies inside the first GroupQueryAttention with a raw access violation (E17..E21, all
at the same point; do_rotary, rotary_interleaved and the rope-cache row count each proven irrelevant
on that topology). The one geometry number no compiling graph in this project has ever used is
kv_num_heads 2 together with head_dim 128: Qwen3-0.6B compiles with kv 8 x 128 (K/V 1024 wide ==
hidden_size) and the vendor's Qwen2-0.5B ships kv 2 x 64 (K/V 128 wide). ERNIE native is the only
16Q / 2KV / 128 combination tried, i.e. the only one where K and V are narrower than hidden_size.

tile_kv_heads.py solved that with Split+Concat, which is now known to be unusable. This script solves
it inside the projection instead: each of the two KV heads' 128 weight columns is repeated four times
in the qkv MatMul weight, so the MatMul becomes 4096 wide and the existing 3-way Split, with sizes
[2048,1024,1024], produces Q 2048 and K/V 1024 -- Qwen3's exact shapes -- with no new nodes at all.
The round trips stay; only their K width (256 -> 1024) changes.

Semantics: icraft maps Q head h to KV head h*kv/num_q, so with the true kv=2 that is h//8. Ordering
the 8 new heads as [o0,o0,o0,o0,o1,o1,o1,o1], Q head h -> new head h//2 -> original (h//2)//4 = h//8
for every h in 0..15; that identity is checked for all 16 heads before anything is written. V
replicates the same way, so the attention output is identical; present.* comes back with each KV head
repeated 4x, which is what the next step feeds back in as past_key_values. logits unchanged.

Acceptance is byte-level and reads both files back: every one of the other 113 external tensors must
be byte-identical at its new offset, and for each of the 18 qkv weights the Q columns must match the
source and each new KV head must equal, column for column, the source head h//rep. (An earlier
version of that check compared new head h against new head h//rep -- self-referential, since new
heads 0..3 are themselves replicas -- and reported a correct file as broken.)

Cost, real and unavoidable if this is the route: the KV cache grows 4x -- 18 layers x 2 tensors x
8 heads x 128 fp16 = 589,824 B per position instead of 147,456 B (at 544 positions, ~40 MB vs ~10 MB;
both inside the board's 256 MB CMA pool, but the driver must be changed to match) -- and the qkv
weights grow 3,145,728 B per layer, 56.6 MB overall.

Mechanics: the whole model.onnx.data is rewritten from scratch, streaming every external initializer
in graph order with the source file's own alignment (measured and printed first), copying bytes for
untouched tensors and the new array for the 18 qkv weights. One layer's weight is 5.2 MB, so the pass
never holds more than one tensor in RAM.

Two operand forms have to be handled: the Split sizes live as an inline INT64 initializer, while the
Reshape shape operands live as Constant nodes (whose payload is an *attribute* -- NodeProto has no
.value field). Both are shared per role across all 18 layers, which the script re-checks here.

Usage: python -X utf8 replicate_qkv_columns.py --src e_v21 --dst e_v22
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
NP_DT = {onnx.TensorProto.FLOAT16: np.float16, onnx.TensorProto.FLOAT: np.float32,
         onnx.TensorProto.BFLOAT16: np.uint16, onnx.TensorProto.INT64: np.int64,
         onnx.TensorProto.INT32: np.int32}


def get_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a.i
    raise SystemExit('attribute %s missing on %s' % (name, node.name))


def set_attr(node, name, val):
    for a in node.attribute:
        if a.name == name:
            old = a.i
            a.i = int(val)
            return old
    raise SystemExit('attribute %s missing on %s' % (name, node.name))


def ext_of(init):
    return {e.key: str(e.value) for e in init.external_data}


def get_dims(vi):
    return [str(d.dim_value if d.HasField('dim_value') else d.dim_param)
            for d in vi.type.tensor_type.shape.dim]


def set_dims(vi, axis, value):
    """Rewrite one shape dim, whether it is stored as dim_value or as a dim_param string."""
    d = vi.type.tensor_type.shape.dim[axis]
    if d.HasField('dim_value'):
        old = str(d.dim_value)
        d.dim_value = int(value)
    else:
        old = d.dim_param
        d.dim_param = str(value)
    return old


def maps(graph):
    bo = {}
    for n in graph.node:
        for o in n.output:
            if o:
                bo[o] = n
    inits = {i.name: i for i in graph.initializer}
    cons = defaultdict(list)
    for n in graph.node:
        for ip in n.input:
            cons[ip].append(n)
    return bo, inits, cons


def operand_ints(name, bo, inits):
    """Read an int64 shape/sizes operand back as a flat list, Constant node or initializer."""
    if name in inits:
        return onnx.numpy_helper.to_array(inits[name]).reshape(-1).tolist()
    p = bo.get(name)
    if p is not None and p.op_type == 'Constant':
        for a in p.attribute:
            if a.name == 'value':
                return onnx.numpy_helper.to_array(a.t).reshape(-1).tolist()
        raise SystemExit('Constant %s has no value attribute' % name)
    raise SystemExit('operand %s is neither an initializer nor a Constant node (%s)'
                     % (name, p and p.op_type))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--dry', action='store_true', help='plan and verify only; write nothing')
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    src_data = os.path.join(src_dir, 'model.onnx.data')
    dst_data = os.path.join(dst_dir, 'model.onnx.data')

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    bo, inits, cons = maps(g)
    if len(inits) != len(g.initializer):
        raise SystemExit('duplicate initializer names')
    out_counts = Counter(o for n in g.node for o in n.output if o)
    dup = [o for o, c in out_counts.items() if c > 1]
    if dup:
        raise SystemExit('tensor produced twice: %s' % sorted(dup)[:5])
    vi_by_name = {}
    for v in list(g.input) + list(g.output) + list(g.value_info):
        vi_by_name.setdefault(v.name, []).append(v)

    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    if len(gqas) != 18:
        raise SystemExit('%d GQA nodes, expected 18' % len(gqas))
    nh = get_attr(gqas[0], 'num_heads')
    knh = get_attr(gqas[0], 'kv_num_heads')
    if nh % knh:
        raise SystemExit('num_heads %d not a multiple of kv_num_heads %d' % (nh, knh))

    # ---- walk each layer's attention front end and record what has to change ----
    plan = []
    head_dim = None
    hidden = None
    for idx, node in enumerate(gqas):
        def op(name, want, what):
            p = bo.get(name)
            if p is None or p.op_type != want:
                raise SystemExit('layer %d %s: expected %s, got %s'
                                 % (idx, what, want, p and p.op_type))
            return p

        r2q = op(node.input[0], 'Reshape', 'slot 0')
        r1q = op(r2q.input[0], 'Reshape', 'q round trip inner')
        split = op(r1q.input[0], 'Split', 'q round trip source')
        r2k = op(node.input[1], 'Reshape', 'slot 1')
        r1k = op(r2k.input[0], 'Reshape', 'k round trip inner')
        if bo.get(r1k.input[0]) is not split:
            raise SystemExit('layer %d: k round trip not fed by the same Split' % idx)
        if split.output[2] != node.input[2]:
            raise SystemExit('layer %d: V slot %s is not Split output_2 %s'
                             % (idx, node.input[2], split.output[2]))
        mm = op(split.input[0], 'MatMul', 'Split source')
        w_name = mm.input[1]
        if w_name not in inits:
            raise SystemExit('layer %d: qkv weight %s is not an initializer' % (idx, w_name))
        w = inits[w_name]
        if w.data_type != onnx.TensorProto.FLOAT16:
            raise SystemExit('layer %d: qkv weight dtype %s, this script only does fp16'
                             % (idx, onnx.TensorProto.DataType.Name(w.data_type)))
        if w.data_location != onnx.TensorProto.EXTERNAL:
            raise SystemExit('layer %d: qkv weight is inline, not external' % idx)
        rows = list(w.dims)
        sv = operand_ints(split.input[1], bo, inits)
        kv_shape = operand_ints(r2k.input[1], bo, inits)
        q_inner = operand_ints(r1q.input[1], bo, inits)
        k_inner = operand_ints(r1k.input[1], bo, inits)
        if head_dim is None:
            head_dim = q_inner[-1]
            if k_inner[-1] != head_dim:
                raise SystemExit('layer %d: inner reshapes disagree (%s vs %s)'
                                 % (idx, q_inner, k_inner))
            if head_dim != sv[1] // knh or head_dim != sv[2] // knh:
                raise SystemExit('layer %d: head_dim %d disagrees with Split sizes %s / kv %d'
                                 % (idx, head_dim, sv, knh))
        elif (q_inner[-1], k_inner[-1]) != (head_dim, head_dim):
            raise SystemExit('layer %d: inner reshape width != head_dim %d' % (idx, head_dim))
        if sv != [nh * head_dim, knh * head_dim, knh * head_dim]:
            raise SystemExit('layer %d: Split sizes %s are not the head geometry' % (idx, sv))
        if rows[1] != sum(sv):
            raise SystemExit('layer %d: weight dims %s vs Split total %d' % (idx, rows, sum(sv)))
        if hidden is None:
            hidden = rows[0]
        elif rows[0] != hidden:
            raise SystemExit('layer %d: weight rows %d != %d' % (idx, rows[0], hidden))
        if kv_shape != [0, -1, sv[1]]:
            raise SystemExit('layer %d: K outer reshape %s does not match the K block width %s'
                             % (idx, kv_shape, sv))
        for nm, want in ((split.input[1], 18), (r2k.input[1], 18)):
            if len(cons[nm]) != want:
                raise SystemExit('layer %d: shared operand %s has %d consumers, expected %d'
                                 % (idx, nm, len(cons[nm]), want))
            if {n.op_type for n in cons[nm]} not in ({'Split'}, {'Reshape'}):
                raise SystemExit('shared operand %s feeds %s' % (nm, {n.op_type for n in cons[nm]}))
        plan.append(dict(idx=idx, w=w_name, rows=rows, q=sv[0], kw=sv[1], vw=sv[2],
                         split=split, sizes_name=split.input[1], kshape_name=r2k.input[1],
                         r2k_out=r2k.output[0]))

    hd = head_dim
    # the target is Qwen3's geometry: K and V each hidden_size wide, i.e. hidden/hd heads
    new_knh = hidden // hd
    if new_knh * hd != hidden:
        raise SystemExit('hidden %d is not a whole number of %d-wide heads' % (hidden, hd))
    if new_knh % knh:
        raise SystemExit('target kv heads %d is not a whole replication of %d' % (new_knh, knh))
    if nh % new_knh:
        raise SystemExit('%d q heads does not divide evenly into %d kv heads' % (nh, new_knh))
    rep = new_knh // knh
    # the whole point: Q head h must still land on the same original KV head as before
    for h in range(nh):
        after = (h * new_knh // nh) // rep
        before = h * knh // nh
        if after != before:
            raise SystemExit('Q head %d: replicated graph reads original head %d, was %d'
                             % (h, after, before))
    print('head mapping preserved for all %d q heads: h -> new %d -> original %d'
          % (nh, nh * new_knh // nh // rep, nh * knh // nh))
    new_q = nh * hd
    new_kw = new_knh * hd
    print('head_dim %d; q %d; kv %d x %d -> %d x %d (rep %d); matmul width %d -> %d; layers %d'
          % (hd, new_q, knh, hd, new_knh, hd, rep, new_q + 2 * knh * hd, new_q + 2 * new_kw,
             len(plan)))

    # the two operands that change are shared per role, so rename each exactly once
    sizes_names = {p['sizes_name'] for p in plan}
    kshape_names = {p['kshape_name'] for p in plan}
    print('distinct sizes operands: %s; distinct K-outer operands: %s'
          % (sorted(sizes_names), sorted(kshape_names)))

    # ---- measure the source data file's own alignment before reproducing it ----
    ext = [i for i in g.initializer if i.data_location == onnx.TensorProto.EXTERNAL]
    offs = [int(ext_of(i).get('offset', 0)) for i in ext]
    src_ext = {i.name: (int(ext_of(i).get('offset', 0)), int(ext_of(i).get('length', 0)))
               for i in ext}
    align = next((a for a in (4096, 256, 128, 64, 32, 16, 8, 4, 2, 1)
                  if all(o % a == 0 for o in offs)), 0)
    print('source external offsets: %d tensors, all multiples of %d' % (len(offs), align))
    if align < 64:
        raise SystemExit('source data is not 64-byte aligned; revisit before rewriting it')
    if len({i.name for i in ext}) != len(ext):
        raise SystemExit('duplicate external initializer names')

    per_layer_extra = 2 * (new_kw - plan[0]['kw']) * plan[0]['rows'][0] * 2
    print('expected data delta = 2 blocks x %d extra cols x %d rows x 2 B x 18 layers = %d B'
          % (new_kw - plan[0]['kw'], plan[0]['rows'][0], per_layer_extra * len(plan)))
    if args.dry:
        print('DRY: nothing written')
        return

    os.makedirs(dst_dir, exist_ok=True)

    # ---- stream the new data file ----
    targets = {p['w']: p for p in plan}
    out_off = 0
    n_streamed = n_replaced = 0
    with open(src_data, 'rb') as fin, open(dst_data, 'wb') as fout:
        for init in g.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL:
                continue
            kv = ext_of(init)
            off = int(kv.get('offset', 0))
            ln = int(kv.get('length', 0))
            loc = kv.get('location', 'model.onnx.data')
            if loc != 'model.onnx.data':
                raise SystemExit('unexpected external location %s for %s' % (loc, init.name))
            if init.name in targets:
                p = targets[init.name]
                fin.seek(off)
                raw = fin.read(ln)
                if len(raw) != ln:
                    raise SystemExit('%s: short read %d of %d' % (init.name, len(raw), ln))
                arr = np.frombuffer(raw, dtype=NP_DT[init.data_type]).reshape(p['rows'])
                if ln != arr.nbytes:
                    raise SystemExit('%s: %d bytes vs dims %s = %d'
                                     % (init.name, ln, p['rows'], arr.nbytes))
                q = arr[:, : p['q']]
                src_blocks = [arr[:, p['q']: p['q'] + p['kw']],
                              arr[:, p['q'] + p['kw']: p['q'] + p['kw'] + p['vw']]]
                blocks = [np.repeat(b.reshape(p['rows'][0], knh, hd), rep, axis=1)
                          .reshape(p['rows'][0], new_kw) for b in src_blocks]
                new = np.ascontiguousarray(np.concatenate([q] + blocks, axis=1))
                # every new KV head must be, column for column, the original head it came from
                for slot, (src_blk, got) in enumerate(zip(src_blocks, blocks)):
                    ref = np.repeat(src_blk.reshape(p['rows'][0], knh, hd), rep, axis=1)
                    ref = ref.reshape(p['rows'][0], new_kw)
                    if not np.array_equal(got, ref):
                        raise SystemExit('layer %d: slot %d replication mismatch' % (p['idx'], slot))
                    for h in range(new_knh):
                        o = h // rep
                        if not np.array_equal(got[:, h * hd: (h + 1) * hd],
                                              src_blk[:, o * hd: (o + 1) * hd]):
                            raise SystemExit('layer %d slot %d: new head %d != original head %d'
                                             % (p['idx'], slot, h, o))
                if not np.array_equal(new[:, : p['q']], q):
                    raise SystemExit('layer %d: Q columns moved' % p['idx'])
                payload = new.tobytes()
                dims = [p['rows'][0], new.shape[1]]
                n_replaced += 1
            else:
                fin.seek(off)
                payload = fin.read(ln)
                dims = list(init.dims)
            pad = (-out_off) % align
            fout.write(b'\x00' * pad)
            out_off += pad
            fout.write(payload)
            init.ClearField('external_data')
            for k, val in (('location', 'model.onnx.data'),
                           ('offset', str(out_off)), ('length', str(len(payload)))):
                e = init.external_data.add()
                e.key = k
                e.value = val
            init.data_location = onnx.TensorProto.EXTERNAL
            del init.dims[:]
            init.dims.extend(dims)
            out_off += len(payload)
            n_streamed += 1
    if n_replaced != len(plan):
        raise SystemExit('replaced %d qkv weights, expected %d' % (n_replaced, len(plan)))
    print('external initializers streamed: %d (%d qkv weights replaced)' % (n_streamed, n_replaced))
    print('new data %d bytes; source %d; actual delta %d'
          % (out_off, os.path.getsize(src_data), out_off - os.path.getsize(src_data)))

    # ---- rewrite the two INT64 operands (one shared node/initializer each) ----
    renames = {}

    def rename_operand(old_name, new_values):
        new_name = '/model/constants/INT64/[%s]' % ', '.join(str(x) for x in new_values)
        if old_name in renames:
            if renames[old_name] != new_name:
                raise SystemExit('shared operand %s asked to become two things' % old_name)
            return
        arr = np.array(new_values, dtype=np.int64)
        if old_name in inits:
            i = inits[old_name]
            i.CopyFrom(onnx.numpy_helper.from_array(arr, name=new_name))
        elif old_name in bo and bo[old_name].op_type == 'Constant':
            c = bo[old_name]
            c.name = '/model/constant_nodes/INT64/[%s]' % ', '.join(str(x) for x in new_values)
            c.output[0] = new_name
            for a in c.attribute:
                if a.name == 'value':
                    a.t.CopyFrom(onnx.numpy_helper.from_array(arr, name=new_name))
                    break
            else:
                raise SystemExit('Constant %s has no value attribute' % old_name)
            bo[new_name] = c
        else:
            raise SystemExit('operand %s is neither initializer nor Constant' % old_name)
        renames[old_name] = new_name

    for p in plan:
        rename_operand(p['sizes_name'], [new_q, new_kw, new_kw])
        rename_operand(p['kshape_name'], [0, -1, new_kw])
    for n in g.node:
        for j, ip in enumerate(n.input):
            if ip in renames:
                n.input[j] = renames[ip]
    for name, new_name in renames.items():
        for v in vi_by_name.get(name, []):
            v.name = new_name
    print('operands renamed: %s' % sorted(renames.items()))

    # ---- shape metadata ----
    fixed = Counter()
    for p in plan:
        i = p['idx']
        for v in vi_by_name.get(p['w'], []):
            d = get_dims(v)
            if d[-1] != str(p['rows'][1]):
                raise SystemExit('layer %d weight value_info %s: last dim %s, expected %s'
                                 % (i, d, p['rows'][1]))
            set_dims(v, len(d) - 1, new_q + 2 * new_kw)
            fixed['weight value_info'] += 1
        for name, old_last, new_last in (
                (p['split'].input[0], p['q'] + p['kw'] + p['vw'], new_q + 2 * new_kw),
                (p['split'].output[1], p['kw'], new_kw),
                (p['split'].output[2], p['vw'], new_kw),
                (p['r2k_out'], p['kw'], new_kw)):
            for v in vi_by_name.get(name, []):
                d = get_dims(v)
                if d[-1] != str(old_last):
                    raise SystemExit('layer %d %s: last dim %s, expected %s'
                                     % (i, name, d[-1], old_last))
                set_dims(v, len(d) - 1, new_last)
                fixed['tensor last dim'] += 1
    print('metadata rewrites: %s' % dict(fixed))

    for node in gqas:
        old = set_attr(node, 'kv_num_heads', new_knh)
        if old != knh:
            raise SystemExit('kv_num_heads was %s, expected %s' % (old, knh))

    touched = 0
    for v in list(g.input) + list(g.output):
        if v.name.startswith('past_key_values.') or v.name.startswith('present.'):
            d = get_dims(v)
            if d[1] != str(knh):
                raise SystemExit('%s axis 1 is %s, expected %s' % (v.name, d[1], knh))
            set_dims(v, 1, new_knh)
            touched += 1
    print('cache tensors rewritten: %d' % touched)
    if touched != 2 * 2 * len(gqas):
        raise SystemExit('expected %d cache tensors, got %d' % (2 * 2 * len(gqas), touched))

    onnx.save(m, os.path.join(dst_dir, 'model.onnx'))

    # ---- re-read from disk and check every layer ----
    chk = onnx.load(os.path.join(dst_dir, 'model.onnx'), load_external_data=False)
    cg = chk.graph
    cbo, cinits, ccons = maps(cg)
    bad = 0
    want = {'sizes': [new_q, new_kw, new_kw], 'inner': [0, -1, hd],
            'q outer': [0, -1, new_q], 'k outer': [0, -1, new_kw]}
    for node in [n for n in cg.node if n.op_type == 'GroupQueryAttention']:
        i = int(node.name.split('/')[2].split('.')[1])
        r2q = cbo[node.input[0]]
        r1q = cbo[r2q.input[0]]
        sp = cbo[r1q.input[0]]
        r2k = cbo[node.input[1]]
        r1k = cbo[r2k.input[0]]
        mm = cbo[sp.input[0]]
        got = {'sizes': operand_ints(sp.input[1], cbo, cinits),
               'inner': operand_ints(r1q.input[1], cbo, cinits),
               'q outer': operand_ints(r2q.input[1], cbo, cinits),
               'k outer': operand_ints(r2k.input[1], cbo, cinits)}
        if got != want:
            print('layer %d MISMATCH %s vs %s' % (i, got, want))
            bad += 1
        if operand_ints(r1k.input[1], cbo, cinits) != [0, -1, hd]:
            print('layer %d K inner %s' % (i, operand_ints(r1k.input[1], cbo, cinits)))
            bad += 1
        w = cinits[mm.input[1]]
        if list(w.dims) != [1024, new_q + 2 * new_kw]:
            print('layer %d weight dims %s' % (i, list(w.dims)))
            bad += 1
        if get_attr(node, 'kv_num_heads') != new_knh or get_attr(node, 'num_heads') != nh:
            print('layer %d attrs %s' % (i, [get_attr(node, a) for a in ('num_heads',
                                                                          'kv_num_heads')]))
            bad += 1
    for v in cg.input:
        if v.name.startswith('past_key_values.') and get_dims(v)[1] != str(new_knh):
            print('%s axis1 %s' % (v.name, get_dims(v)))
            bad += 1
    for v in cg.output:
        if v.name.startswith('present.') and get_dims(v)[1] != str(new_knh):
            print('%s axis1 %s' % (v.name, get_dims(v)))
            bad += 1
    print('per-layer re-read mismatches: %d' % bad)
    if bad:
        raise SystemExit('refusing to leave a broken fmodel')
    names = set(cbo) | set(cinits) | {v.name for v in cg.input} | {''}
    dangling = [(n.name, ip) for n in cg.node for ip in n.input if ip not in names]
    print('dangling operands: %d' % len(dangling))
    for d in dangling[:10]:
        print('  ', d)
    if dangling:
        raise SystemExit('refusing to leave dangling operands')
    # byte-level acceptance: re-read both files and compare every tensor where it sits now
    targets_by_name = {p['w']: p for p in plan}
    carried = checked = 0
    with open(dst_data, 'rb') as f, open(src_data, 'rb') as sf:
        for w in cg.initializer:
            if w.data_location != onnx.TensorProto.EXTERNAL:
                continue
            kv = ext_of(w)
            off = int(kv['offset'])
            ln = int(kv['length'])
            dims = list(w.dims)
            old_off, old_ln = src_ext[w.name]
            f.seek(off)
            new_bytes = f.read(ln)
            if len(new_bytes) != ln:
                raise SystemExit('%s: short read %d of %d at offset %d' % (w.name, len(new_bytes),
                                                                            ln, off))
            sf.seek(old_off)
            old_bytes = sf.read(old_ln)
            if len(old_bytes) != old_ln:
                raise SystemExit('%s: source short read' % w.name)
            p = targets_by_name.get(w.name)
            if p is None:
                if new_bytes != old_bytes:
                    raise SystemExit('%s: carried-over bytes differ (len %d -> %d at %d -> %d)'
                                     % (w.name, old_ln, ln, old_off, off))
                carried += 1
                continue
            if dims != [p['rows'][0], new_q + 2 * new_kw]:
                raise SystemExit('layer %d: on-disk dims %s' % (p['idx'], dims))
            if ln != dims[0] * dims[1] * 2 or old_ln != p['rows'][0] * sum(
                    [p['q'], p['kw'], p['vw']]) * 2:
                raise SystemExit('layer %d: length %d/%d does not match the geometry'
                                 % (p['idx'], ln, old_ln))
            narr = np.frombuffer(new_bytes, dtype=np.float16).reshape(dims)
            oarr = np.frombuffer(old_bytes, dtype=np.float16).reshape(p['rows'])
            if not np.array_equal(narr[:, : p['q']], oarr[:, : p['q']]):
                raise SystemExit('layer %d: Q columns changed on disk' % p['idx'])
            # each new KV head must be, on disk, the source head it was replicated from
            for slot in (0, 1):
                src_base = p['q'] + slot * p['kw']
                dst_base = p['q'] + slot * new_kw
                for h in range(new_knh):
                    o = h // rep
                    if not np.array_equal(narr[:, dst_base + h * hd: dst_base + (h + 1) * hd],
                                          oarr[:, src_base + o * hd: src_base + (o + 1) * hd]):
                        raise SystemExit('layer %d slot %d: on-disk head %d != source head %d'
                                         % (p['idx'], slot, h, o))
            checked += 1
    print('on-disk: %d tensors carried over byte-identical, %d qkv weights verified '
          'head by head against their source bytes' % (carried, checked))
    if checked != len(plan):
        raise SystemExit('only %d qkv weights verified' % checked)
    if os.path.getsize(dst_data) != out_off:
        raise SystemExit('data file is %d bytes but %d were written'
                         % (os.path.getsize(dst_data), out_off))
    print('onnx %d bytes, data %d bytes'
          % (os.path.getsize(os.path.join(dst_dir, 'model.onnx')), out_off))
    print('DONE_REPLICATION')


if __name__ == '__main__':
    main()
