"""Test whether the parser's decoder-layer recognition needs a per-head QK norm in front of the GQA.

The census that made this worth testing (all from the same .icraft logs):

    every Qwen3-derived build  -> network_input emits 58 tensors (2 + 56 past_key_values)
    every ERNIE-derived build  -> network_input emits  2 tensors (the 36 caches are never registered)

except E10/E16, where the GQA slots were themselves graph inputs and the parser emitted 41/92. So the
cache registration is conditional on something in the graph, and ERNIE's GQA -- which does consume
past_key_values.N.key/value on slots 3/4 -- then dereferences a tensor that was never wired in. That
is the access violation (0xC0000005, not a caught bad_alloc) at the first GQA, and it is why no amount
of attribute, operand-form, node-order or declaration work on the attention subgraph changed anything.

The one remaining structural asymmetry between the two graphs is the per-head norm:

    q3n0   SimplifiedLayerNormalization 55  (model.layers.N.attn.q_norm + k_norm for N = 1..27,
                                             plus layer 0's input_layernorm; layer 0 has no q/k norm)
    ERNIE  SimplifiedLayerNormalization  1  (only layer 0's input_layernorm)

and it is not something my rewrite chain deleted: the pristine onnxruntime-genai export e_exported has
zero q_norm/k_norm weight initializers, because ERNIE-4.5-0.3B-PT does not normalise per head.

Two probes, one per side, so the claim is either confirmed or killed rather than asserted:

  dropnorms  q3n0 -> q3g0: delete every q_norm/k_norm SLN and let the following Reshape_2 read the
             Reshape_1 output directly. All 28 layers then have ERNIE's Split->Reshape->Reshape->GQA
             form. Prediction if the hypothesis is right: network_input collapses to 2 tensors and the
             parse dies at the first GQA exactly like ERNIE. Semantics are changed on purpose; this run
             is judged only on where the parse stops.
  addnorms   e_v35 -> e_v36: insert a q_norm/k_norm SLN between Reshape_1 and Reshape_2 of all 18
             layers, with an inline fp16 ones weight of shape [128] and the attribute triple copied
             from q3n0's own q_norm node. This is a DIAGNOSTIC, not a fix: RMS-normalising each head
             vector is not an identity, so a pass here only proves the parser needs the node -- the
             numerical cost has to be settled separately (weight = ones is not enough).

Both keep every width legal and only touch the graph, never the weights: model.onnx.data is hardlinked.
"""
import argparse
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def bo_of(g):
    bo = {}
    for n in g.node:
        for o in n.output:
            if o:
                bo[o] = n
    return bo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('dropnorms', 'addnorms'))
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)
    m = onnx.load(os.path.join(src, 'model.onnx'), load_external_data=False)
    g = m.graph
    gq = [n for n in g.node if n.op_type == 'GroupQueryAttention']

    if args.mode == 'dropnorms':
        targets = [n for n in g.node
                   if n.op_type == 'SimplifiedLayerNormalization' and '/attn/' in n.name]
        print('per-head SLN found: %d' % len(targets))
        if not targets:
            raise SystemExit('nothing to drop')
        for n in targets:
            consumers = [c for c in g.node if n.output[0] in c.input]
            if len(consumers) != 1 or consumers[0].op_type != 'Reshape':
                raise SystemExit('%s has consumers %s' % (n.name, [c.name for c in consumers]))
            producers = [p for p in g.node if n.input[0] in (p.output or [''])[0:1]]
            if len(producers) != 1 or producers[0].op_type != 'Reshape':
                raise SystemExit('%s producer is %s' % (n.name, [p.name for p in producers]))
            for k, ip in enumerate(consumers[0].input):
                if ip == n.output[0]:
                    consumers[0].input[k] = producers[0].output[0]
                    break
            else:
                raise SystemExit('%s output is not an operand of %s' % (n.name, consumers[0].name))
            for v in list(g.value_info):
                if v.name == n.output[0]:
                    g.value_info.remove(v)
            g.node.remove(n)
        print('dropped; SLN now %d' % sum(1 for n in g.node
                                         if n.op_type == 'SimplifiedLayerNormalization'))

    if args.mode == 'addnorms':
        tmpl = [n for n in onnx.load(os.path.join(ROOT, 'q3n0', 'model.onnx'),
                                     load_external_data=False).graph.node
                if n.op_type == 'SimplifiedLayerNormalization' and 'q_norm' in n.name][0]
        attrs = {a.name: a for a in tmpl.attribute}
        print('template %s attrs %s' % (tmpl.name, {k: (v.i if v.type == 2 else v.f)
                                                    for k, v in attrs.items()}))
        inserted = 0
        for node in gq:
            for slot, tag in ((0, 'q'), (1, 'k')):
                r2 = bo_of(g).get(node.input[slot])
                if r2 is None or r2.op_type != 'Reshape':
                    raise SystemExit('slot %d not a Reshape (%s)' % (slot, node.input[slot]))
                r1 = [p for p in g.node if p.output[0] == r2.input[0]]
                if len(r1) != 1 or r1[0].op_type != 'Reshape':
                    raise SystemExit('slot %d Reshape_2 is not fed by a Reshape_1' % slot)
                layer = node.name.split('/')[2]
                wname = 'model.%s.attn.%s_norm.layernorm.weight' % (layer, tag)
                if any(i.name == wname for i in g.initializer):
                    raise SystemExit('%s already exists' % wname)
                wt = onnx.numpy_helper.from_array(np.ones(128, dtype=np.float16), wname)
                g.initializer.append(wt)
                out = '/model/%s/attn/%s_norm/SimplifiedLayerNormalization/output_0' % (layer, tag)
                s = onnx.helper.make_node('SimplifiedLayerNormalization',
                                          [r1[0].output[0], wname], [out],
                                          name='/model/%s/attn/%s_norm/'
                                               'SimplifiedLayerNormalization' % (layer, tag))
                for a in attrs.values():
                    s.attribute.append(onnx.helper.make_attribute(a.name,
                                                                  a.i if a.type == 2 else a.f))
                g.node.insert(list(g.node).index(r1[0]) + 1, s)
                r2.input[0] = out
                vis = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
                if r1[0].output[0] in vis:
                    nv = onnx.helper.make_tensor_value_info(
                        out, vis[r1[0].output[0]].type.tensor_type.elem_type,
                        [d.dim_value if d.HasField('dim_value') else d.dim_param
                         for d in vis[r1[0].output[0]].type.tensor_type.shape.dim])
                    g.value_info.append(nv)
                inserted += 1
        print('SLN inserted: %d; SLN total now %d' % (inserted, sum(
            1 for n in g.node if n.op_type == 'SimplifiedLayerNormalization')))

    names = ({o for n in g.node for o in n.output if o}
             | {i.name for i in g.initializer} | {v.name for v in g.input} | {''})
    dangling = [(n.name, ip) for n in g.node for ip in n.input if ip not in names]
    print('nodes %d; dangling operands %d' % (len(g.node), len(dangling)))
    for d in dangling[:5]:
        print('  ', d)
    if dangling:
        raise SystemExit('refusing to save a graph with dangling operands')

    onnx.save(m, os.path.join(dst, 'model.onnx'))
    s = os.path.join(src, 'model.onnx.data')
    t = os.path.join(dst, 'model.onnx.data')
    if os.path.exists(s):
        if os.path.exists(t):
            os.remove(t)
        os.link(s, t)
        print('linked %s (%d bytes)' % (t, os.path.getsize(t)))

    chk = onnx.load(os.path.join(dst, 'model.onnx'), load_external_data=False).graph
    b2 = bo_of(chk)
    n0 = [n for n in chk.node if n.op_type == 'GroupQueryAttention'][0]
    for slot in (0, 1):
        chain = []
        cur = n0.input[slot]
        while cur in b2 and len(chain) < 4:
            chain.append(b2[cur].op_type)
            cur = b2[cur].input[0]
        print('slot%d chain from GQA: %s' % (slot, ' <- '.join(chain)))
    print('SLN=%d SkipSLN=%d' % (sum(1 for n in chk.node if n.op_type == 'SimplifiedLayerNormalization'),
                                 sum(1 for n in chk.node
                                     if n.op_type == 'SkipSimplifiedLayerNormalization')))
    print('DONE_%s' % args.mode.upper())


if __name__ == '__main__':
    main()
