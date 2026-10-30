"""Crop a decoder graph to its first N layers, keeping names and indices contiguous.

Why this specific cut. The census over every parse log in .icraft/logs is one clean binary split:

    Qwen3-derived builds  -> network_input emits 58 tensors, parse succeeds (Q3N 1770 ops, Q3G 1715)
    ERNIE-derived builds  -> network_input emits  2 tensors, parse dies entering the first GQA

and the two-sided norm test just killed the per-head-norm explanation from both directions: Q3G is
Qwen3 with all 54 q_norm/k_norm SLN deleted (28 layers of exactly ERNIE's Split->Reshape->Reshape->GQA
form) and it parses, while E36 is ERNIE with 36 identity-weight SLN inserted and still emits 2 network
tensors and dies at the same op. So the accepted chain form, the operand forms, the declarations, the
rope-cache geometry and the norms are all excluded. The last structural axis that still separates the
two graphs is the number of decoder layers: 28 vs 18.

Crop the *parsing* graph to 18 layers. Both outcomes are informative:
  parses  -> layer count is irrelevant; what is left between ERNIE and Qwen3 is only vocab 103424,
             epsilon 1e-5 and the weight values themselves, so the next probe has to be data-side.
  crashes -> the parser's cache registration is layer-count dependent and ERNIE's graph form was never
             the problem.

Layers are deleted from the top (index >= N) so the surviving indices stay 0..N-1 and no renaming is
needed. Removed: every node named /model/layers.K/..., the past_key_values.K.* graph inputs, the
present.K.* graph outputs, their value_info entries, and the model.layers.K.* initializers (dropping an
external-data initializer leaves the hardlinked .data untouched, which is what we want).

Usage: python -X utf8 crop_layers.py --src q3g0 --dst q3h0 --keep 18
"""
import argparse
import os
import re
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--keep', type=int, required=True)
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)
    m = onnx.load(os.path.join(src, 'model.onnx'), load_external_data=False)
    g = m.graph
    lay = re.compile(r'(?:layers|present|past_key_values)\.(\d+)')

    def idx_of(name):
        mm = lay.search(name or '')
        return int(mm.group(1)) if mm else None

    def is_final(tag):
        # the exporter numbers the trailing norm as layers.<num_layers>, so it is not a decoder layer
        return 'final_norm' in (tag or '')

    drop_nodes = [n for n in g.node
                  if not is_final(n.name) and not is_final(n.output[0])
                  and ((idx_of(n.name) or -1) >= args.keep
                       or (idx_of(n.output[0]) or -1) >= args.keep)]
    kept = [n for n in g.node if n not in drop_nodes]
    print('nodes %d -> %d (dropping %d from layers >= %d)'
          % (len(g.node), len(kept), len(drop_nodes), args.keep))
    if not drop_nodes:
        raise SystemExit('nothing to crop')
    del g.node[:]
    g.node.extend(kept)

    def prune(seq, what):
        before = len(seq)
        for v in list(seq):
            if not is_final(v.name) and (idx_of(v.name) or -1) >= args.keep:
                seq.remove(v)
        print('%s %d -> %d' % (what, before, len(seq)))
    prune(g.input, 'graph inputs')
    prune(g.output, 'graph outputs')
    prune(g.value_info, 'value_info')
    before = len(g.initializer)
    for i in list(g.initializer):
        if not is_final(i.name) and (idx_of(i.name) or -1) >= args.keep:
            g.initializer.remove(i)
    print('initializers %d -> %d' % (before, len(g.initializer)))

    producers = {o for n in g.node for o in n.output} | {i.name for i in g.initializer}
    declared = {v.name for v in g.input}
    # the trailing SkipLayerNorm is fed by the deleted top layer's residual carry and mlp output, so
    # point those operands at the surviving last layer (same shapes, both are the hidden stream)
    def retarget(name):
        if idx_of(name) is None or idx_of(name) < args.keep:
            return name
        return lay.sub('layers.%d' % (args.keep - 1), name, count=1)
    for n in g.node:
        for k, ip in enumerate(n.input):
            if ip and ip not in producers and ip not in declared:
                cand = retarget(ip)
                if cand in producers or cand in declared:
                    print('   rewire %s: %s -> %s' % (n.name, ip, cand))
                    n.input[k] = cand
    dangling = [(n.name, ip) for n in g.node for ip in n.input
                if ip and ip not in producers and ip not in declared]
    gqa = [n.name for n in g.node if n.op_type == 'GroupQueryAttention']
    if len(gqa) != args.keep:
        raise SystemExit('%d GQA nodes, expected %d' % (len(gqa), args.keep))
    print('GQA %d; dangling operands %d' % (len(gqa), len(dangling)))
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
    print('DONE_CROP%d' % args.keep)


if __name__ == '__main__':
    main()
