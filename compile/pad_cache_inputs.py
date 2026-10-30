"""Pad a cropped graph with unused cache-input declarations, to split two causes the crop confounds.

The bracket on q3g0 says the parser's registration of past_key_values.* inputs switches on something
about the *cropped* graph, not on my serialization: q3g0_rt (28 layers, round-tripped through onnx.save
with no structural change) parses with 58 network tensors, and the 27-layer crop parses with 56, while
the 18-layer crop registers only 2 and dies entering the first GroupQueryAttention. Two quantities
moved together in that last step: the number of GroupQueryAttention nodes (28 -> 18) and the number of
cache input declarations (56 -> 36).

This script holds the layer count at --keep and pushes the *declaration* count back up by appending
graph inputs named past_key_values.N.key / .value for N = keep..27, byte-for-byte the same declared
type and symbolic shape as the surviving ones, consumed by nothing. Readings:

  registers the padded count -> registration is declaration-count driven, so an 18-layer model can be
      rescued by padding its cache declarations (and the GQA crash is a separate, downstream question).
  still registers 2          -> registration is keyed on the layer/GQA count and padding is useless;
      the only route left for an 18-layer model is a different graph shape.

Usage: python -X utf8 pad_cache_inputs.py --src q3h0 --dst q3h0_p --from 18 --to 28
"""
import argparse
import os
import re
import sys

import onnx
from onnx import helper

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
LAY = re.compile(r'(?:layers|present|past_key_values)\.(\d+)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--from', dest='lo', type=int, required=True, help='first padded layer index')
    ap.add_argument('--to', dest='hi', type=int, required=True, help='one past the last padded index')
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)
    m = onnx.load(os.path.join(src, 'model.onnx'), load_external_data=False)
    g = m.graph

    tpl = [v for v in g.input if v.name.startswith('past_key_values.')]
    if not tpl:
        raise SystemExit('%s declares no past_key_values inputs' % args.src)
    by_suffix = {}
    for v in tpl:
        sfx = v.name.rsplit('.', 1)[1]
        by_suffix.setdefault(sfx, v)
    print('templates: %s' % {k: (k, [d.dim_value if d.HasField('dim_value') else d.dim_param
                                     for d in list(by_suffix[k].type.tensor_type.shape.dim)])
                            for k in by_suffix})

    declared = {v.name for v in g.input}
    added = []
    for idx in range(args.lo, args.hi):
        for sfx, proto in sorted(by_suffix.items()):
            name = 'past_key_values.%d.%s' % (idx, sfx)
            if name in declared:
                print('   already declared, skip %s' % name)
                continue
            dims = [d.dim_value if d.HasField('dim_value') else d.dim_param
                    for d in list(proto.type.tensor_type.shape.dim)]
            t = helper.make_tensor_value_info(name, proto.type.tensor_type.elem_type, dims)
            g.input.append(t)
            declared.add(name)
            added.append(name)
    print('%s -> %s: graph inputs %d -> %d (added %d unconsumed cache decls %d..%d)'
          % (args.src, args.dst, len(tpl) + 2, len(g.input), len(added), args.lo, args.hi - 1))
    if not added:
        raise SystemExit('nothing padded')

    consumers = [n.name for n in g.node for ip in n.input if ip in set(added)]
    print('consumers of the padded inputs: %d (expected 0)' % len(consumers))
    onnx.save(m, os.path.join(dst, 'model.onnx'))
    s = os.path.join(src, 'model.onnx.data')
    t = os.path.join(dst, 'model.onnx.data')
    if os.path.exists(s):
        if os.path.exists(t):
            os.remove(t)
        os.link(s, t)
    chk = onnx.load(os.path.join(dst, 'model.onnx'), load_external_data=False)
    names = [v.name for v in chk.graph.input]
    print('reloaded inputs %d; past_key_values %d; dups %s'
          % (len(names), len([x for x in names if x.startswith('past_key_values.')]),
             len(names) - len(set(names))))
    print('DONE_PAD %s' % args.dst)


if __name__ == '__main__':
    main()
