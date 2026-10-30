"""Do the Splits I inserted give icraft duplicate tensor names that its own exporter never produces?

split_qkv.py / tile_kv_heads.py each build one INT64 'sizes' initializer per layer (or one shared
object reused 36 times). ONNX allows both, but a parser that keys its tensor table by name could
either de-dup harmlessly or walk off the end. Compare the name census against q3i_v1, which parses.
"""
import os
import sys
from collections import Counter

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork')
DEFAULTS = [
    os.path.join(ROOT, 'qwen3', 'compile', 'fmodel', 'q3i_v1', 'model.onnx'),
    os.path.join(ROOT, 'ernie', 'compile', 'fmodel', 'e_v7', 'model.onnx'),
]


def main():
    paths = sys.argv[1:] or DEFAULTS
    for p in paths:
        m = onnx.load(p, load_external_data=False)
        g = m.graph
        init = Counter(i.name for i in g.initializer)
        nodes = Counter(n.name for n in g.node)
        # every tensor produced more than once (illegal graph)
        outs = Counter(o for n in g.node for o in n.output)
        dup_init = {k: v for k, v in init.items() if v > 1}
        dup_node = {k: v for k, v in nodes.items() if v > 1}
        dup_out = {k: v for k, v in outs.items() if v > 1}
        print('=' * 70)
        print('%s  nodes=%d inits=%d' % (os.path.basename(os.path.dirname(p)),
                                        len(g.node), len(g.initializer)))
        print('  dup initializer names: %d %s' % (len(dup_init), list(dup_init.items())[:4]))
        print('  dup node names       : %d %s' % (len(dup_node), list(dup_node.items())[:4]))
        print('  dup tensor outputs   : %d %s' % (len(dup_out), list(dup_out.items())[:4]))
        # initializers whose name collides with a graph input or value_info name
        names_in = {v.name for v in g.input}
        names_vi = {v.name for v in g.value_info}
        print('  init names that are also graph inputs   : %d' %
              len([k for k in init if k in names_in]))
        print('  value_info names that are also inits    : %d' %
              len([k for k in names_vi if k in init]))
        # how many Split nodes reference the same sizes initializer, and does it exist once
        for n in g.node:
            if n.op_type == 'Split' and len(n.input) == 2:
                pass
        by = Counter(n.input[1] for n in g.node if n.op_type == 'Split' and len(n.input) == 2)
        print('  Split sizes initializers used by >1 node: %s' %
              [(k, v) for k, v in by.items() if v > 1][:3])
        print('  distinct sizes initializers             : %d (for %d Splits)' %
              (len(by), sum(by.values())))
        print('  Constant nodes in graph                 : %d  %s' % (
            sum(1 for n in g.node if n.op_type == 'Constant'),
            sorted({n.output[0] for n in g.node if n.op_type == 'Constant'})[:4]))
        for k in list(by)[:2]:
            i = [x for x in g.initializer if x.name == k]
            print('    %r present %d times in g.initializer dtype=%s dims=%s, producers=%s' % (
                k, len(i),
                onnx.TensorProto.DataType.Name(i[0].data_type) if i else 'ABSENT',
                list(i[0].dims) if i else '-',
                [n.op_type + ':' + n.name for n in g.node if k in list(n.output)][:2] or 'none'))


if __name__ == '__main__':
    main()
