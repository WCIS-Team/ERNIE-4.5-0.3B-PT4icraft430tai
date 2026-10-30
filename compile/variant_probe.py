"""Parse-probe variants of an fmodel dir that change only the graph proto, never the data.

e_v22's layer-0 GQA is now field-for-field identical to the Qwen3 graph that parses (q3n0) except one
attribute, rotary_interleaved (ERNIE 1, Qwen3 0) -- see dump_gqa_signature.py. And E22's parse log
stops right after q_norm/Reshape_2, which is also exactly where the K round trip would be lowered, so
"the GQA crashes" and "the K round trip crashes" are still not separated. This script makes both
probes from one base:

  ri0          rotary_interleaved 1 -> 0 on every GQA. Shape-neutral (the rope cache is head_dim/2
               wide either way). If this is what makes it lower, the semantics still have to be
               repaired afterwards by permuting the q/k head columns pair-major -> split-major, which
               is free: the same permutation on q and k leaves every q.k score unchanged.
  knodirect    feed GQA slot 1 straight from Split output_1, deleting that layer's round trip. Keeps
               the Q round trip, so the two probes tell apart the K chain from the GQA node.
  constsizes   put every inline INT64 Split-sizes initializer back into the exporter's own form: a
               Constant node named /model/constant_nodes/INT64/[...] placed immediately before its
               first consumer, output named /model/constants/INT64/[...], the inner tensor carrying
               that same output name, plus the value_info entry the exporter writes for it (elem_type
               INT64, shape with no dims). add_head_roundtrip.py rebuilt the qkv Split with a plain
               initializer instead, which is the one remaining difference between ERNIE's layer-0
               attention and the Qwen3 graph that parses -- see dump_gqa_signature.py. Shape-neutral.
  dims         rewrite every dim_param whose text is all digits into a real dim_value. The Qwen3 graph
               that parses uses exactly five symbolic axes (batch_size, sequence_length,
               past_sequence_length, total_sequence_length, sequence_length * num_attention_heads /
               num_key_value_heads); every literal size is a dim_value. add_head_roundtrip.py wrote the
               value_info of the Reshapes it inserted with the trailing axis as the *string* '128' (and
               '2048'/'256'), so the GQA's q and k slot declarations carry a symbol named "2048" instead
               of the number 2048 -- 72 tensors = 4 per layer x 18. Shape-neutral, semantically equal.
               (E27: still dies at the GQA, so this form is excluded as the cause.)
  constform    make the three shape Constants that add_head_roundtrip.py inserted indistinguishable
               from exporter ones, at all three places where it deviated: the value_info of a Constant
               is written with an *empty* shape (no dim at all) by the exporter, while mine declares
               shape {dim {dim_value: 3}}; the inner tensor of the Constant carries a '/value' suffix
               instead of the node's own output name; and the exporter emits each Constant immediately
               before its first consumer, while mine are hoisted to graph indices 0,1,2. Node-order
               change only for the Constants, no shape or value touched.
  order        put each layer's q round trip before its k round trip, as the exporter orders them (q
               R1, q R2, k R1, k R2, GQA). add_head_roundtrip.py inserted k's pair first, and the E25
               parse log proves the parser lowers them in that order (op 5,6 = k_norm; op 7,8 =
               q_norm) while Q3N lowers q first (op 5,6) then k (7,8). Swapping two adjacent pairs
               that only depend on already-produced tensors; shape-neutral.
  ri0_knodirect both of the earlier probes.

Only the .onnx changes, so the .data is hardlinked; sizes and offsets stay valid.
Usage: python -X utf8 variant_probe.py --src e_v22 --dst e_v25 --mode constsizes
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--mode', required=True,
                    choices=('ri0', 'knodirect', 'ri0_knodirect', 'constsizes',
                             'ri0_constsizes', 'dims', 'constform', 'order'))
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)
    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    bo = {}
    for n in g.node:
        for o in n.output:
            if o:
                bo[o] = n
    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    if len(gqas) != 18:
        raise SystemExit('%d GQA nodes' % len(gqas))

    if args.mode in ('ri0', 'ri0_knodirect', 'ri0_constsizes'):
        flipped = 0
        for node in gqas:
            for a in node.attribute:
                if a.name == 'rotary_interleaved':
                    if a.i not in (0, 1):
                        raise SystemExit('rotary_interleaved is %s, not 0/1' % a.i)
                    a.i = 0 if a.i == 1 else 1
                    flipped += 1
        print('rotary_interleaved flipped on %d GQA nodes' % flipped)

    if args.mode in ('knodirect', 'ri0_knodirect'):
        removed = 0
        for node in gqas:
            r2k = bo.get(node.input[1])
            if r2k is None or r2k.op_type != 'Reshape':
                raise SystemExit('slot 1 is not a Reshape (%s)' % (r2k and r2k.op_type))
            r1k = bo.get(r2k.input[0])
            if r1k is None or r1k.op_type != 'Reshape':
                raise SystemExit('K round trip inner is not a Reshape')
            split = bo.get(r1k.input[0])
            if split is None or split.op_type != 'Split':
                raise SystemExit('K round trip does not start at the qkv Split')
            consumers = [n for n in g.node if r1k.output[0] in n.input and n is not r2k]
            if consumers:
                raise SystemExit('K round trip inner has other consumers: %s'
                                 % [c.name for c in consumers])
            node.input[1] = split.output[1]
            g.node.remove(r2k)
            g.node.remove(r1k)
            for v in list(g.value_info):
                if v.name in (r1k.output[0], r2k.output[0]):
                    g.value_info.remove(v)
            removed += 1
        print('K round trips removed: %d' % removed)

    if args.mode in ('constsizes', 'ri0_constsizes'):
        made = 0
        for t in [i for i in g.initializer
                  if i.name.startswith('/model/constants/')
                  and i.data_location == onnx.TensorProto.DEFAULT
                  and i.data_type == onnx.TensorProto.INT64]:
            name = t.name
            vals = onnx.numpy_helper.to_array(t).reshape(-1).tolist()
            consumers = [n for n in g.node if name in n.input]
            if not consumers:
                raise SystemExit('%s has no consumers' % name)
            if any(v.name == name for v in g.value_info):
                raise SystemExit('%s already has a value_info' % name)
            pos = min(list(g.node).index(c) for c in consumers)
            node = onnx.helper.make_node(
                'Constant', [], [name],
                name='/model/constant_nodes/INT64/[%s]' % ', '.join(str(x) for x in vals),
                value=onnx.helper.make_tensor(name, onnx.TensorProto.INT64, [len(vals)], vals))
            g.initializer.remove(t)
            g.node.insert(pos, node)
            g.value_info.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.INT64, None))
            print('  %s -> Constant node at %d (before %s), value_info added'
                  % (name, pos, consumers[0].name))
            made += 1
        print('inline INT64 constants converted to Constant nodes: %d' % made)
        if not made:
            raise SystemExit('nothing to convert: the graph is already in the exporter form')

    if args.mode == 'dims':
        import re
        num = re.compile(r'^\d+$')
        fixed = 0
        touched = set()
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField('dim_param') and num.match(d.dim_param):
                    n = int(d.dim_param)
                    d.ClearField('dim_param')
                    d.dim_value = n
                    fixed += 1
                    touched.add(vi.name)
            if vi.name.startswith('/model/constants/'):
                print('  value_info %s dims %s'
                      % (vi.name, [(dd.dim_value if dd.HasField('dim_value') else dd.dim_param)
                                   for dd in vi.type.tensor_type.shape.dim]))
        print('numeric dim_params converted to dim_values: %d across %d tensors'
              % (fixed, len(touched)))
        if not fixed:
            raise SystemExit('nothing to convert')

    if args.mode == 'constform':
        vis = {v.name: v for v in g.value_info}
        renam = rank = moved = 0
        for n in [x for x in g.node if x.op_type == 'Constant'
                  and x.output[0].startswith('/model/constants/')]:
            out = n.output[0]
            if n.attribute[0].t.name != out:
                n.attribute[0].t.name = out
                renam += 1
            vi = vis.get(out)
            if vi is not None and len(vi.type.tensor_type.shape.dim):
                del vi.type.tensor_type.shape.dim[:]
                rank += 1
            cons = [x for x in g.node if out in x.input]
            if not cons:
                raise SystemExit('%s unused' % out)
            cur = list(g.node).index(n)
            pos = min(list(g.node).index(c) for c in cons)
            if cur != pos - 1:
                g.node.remove(n)
                pos = min(list(g.node).index(c) for c in cons)
                g.node.insert(pos, n)
                moved += 1
                print('  %s moved %d -> %d (first consumer %s)'
                      % (out, cur, pos, cons[0].name))
        print('Constant inner names fixed %d; value_info shapes emptied %d; relocated %d'
              % (renam, rank, moved))

    if args.mode == 'order':
        swapped = 0
        for layer in range(18):
            pref = '/model/layers.%d/attn/' % layer

            def one(nm):
                r = [i for i, n in enumerate(g.node) if n.name == pref + nm]
                if len(r) != 1:
                    raise SystemExit('layer %d: %d nodes named %s' % (layer, len(r), nm))
                return r[0]

            k1, k2 = one('k_norm/Reshape_1'), one('k_norm/Reshape_2')
            q1, q2 = one('q_norm/Reshape_1'), one('q_norm/Reshape_2')
            if (k1, k2, q1, q2) != (min(k1, q1), min(k1, q1) + 1, min(k1, q1) + 2,
                                    min(k1, q1) + 3):
                raise SystemExit('layer %d: the four Reshapes are not one contiguous k,k,q,q block: '
                                 '%s' % (layer, [k1, k2, q1, q2]))
            kn1 = g.node[k1]
            qn1, qn2 = g.node[q1], g.node[q2]
            g.node.remove(qn1)
            g.node.remove(qn2)
            pos = list(g.node).index(kn1)
            g.node.insert(pos, qn1)
            g.node.insert(pos + 1, qn2)
            swapped += 1
            if layer == 0:
                print('  layer 0 order now %s'
                      % [n.name.split('/')[-2] + '/' + n.name.split('/')[-1]
                         for n in list(g.node)[pos:pos + 4]])
        print('q-before-k swaps: %d' % swapped)

    names = ({o for n in g.node for o in n.output if o}
             | {i.name for i in g.initializer} | {v.name for v in g.input} | {''})
    dangling = [(n.name, ip) for n in g.node for ip in n.input if ip not in names]
    print('nodes %d; dangling operands %d' % (len(g.node), len(dangling)))
    for d in dangling[:5]:
        print('  ', d)
    if dangling:
        raise SystemExit('refusing to save a graph with dangling operands')

    onnx.save(m, os.path.join(dst_dir, 'model.onnx'))
    for f in ('model.onnx.data',):
        s = os.path.join(src_dir, f)
        t = os.path.join(dst_dir, f)
        if not os.path.exists(s):
            continue
        if os.path.exists(t):
            os.remove(t)
        os.link(s, t)
        print('linked %s (%d bytes)' % (t, os.path.getsize(t)))
    chk = onnx.load(os.path.join(dst_dir, 'model.onnx'), load_external_data=False)
    n0 = [n for n in chk.graph.node if n.op_type == 'GroupQueryAttention'][0]
    print('layer0 GQA inputs %s' % list(n0.input)[:3])
    print('layer0 GQA attrs %s' % {a.name: (a.i if a.type == 2 else a.f)
                                   for a in n0.attribute})
    print('DONE_%s' % args.mode.upper())


if __name__ == '__main__':
    main()
