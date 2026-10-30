"""Enumerate every symbolic dim_param used in the two graphs, with where it appears.

The ERNIE parse log printed the 4-axis past tensors as [1,2,0] (three axes), which suggests icraft
dropped the axis it could not name rather than resolving it; Qwen3's export has the head dim baked
as a literal 128, so that graph never exercised a symbolic head dim. Compare the two name sets to
see exactly which axes each graph leaves dynamic.
"""
import collections
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork')
GRAPHS = {
    'ERNIE': os.path.join(ROOT, 'ernie', 'compile', 'fmodel', 'e_v1', 'model.onnx'),
    'Qwen3': os.path.join(ROOT, 'qwen3', 'compile', 'fmodel', 'q3i_v1', 'model.onnx'),
}


def main():
    for label, path in GRAPHS.items():
        m = onnx.load(path, load_external_data=False)
        g = m.graph
        uses = collections.Counter()
        shapes = collections.defaultdict(set)
        for coll, tag in ((g.input, 'input'), (g.output, 'output'), (g.value_info, 'value_info')):
            for v in coll:
                tt = v.type.tensor_type
                if not tt.HasField('shape'):
                    continue
                for d in tt.shape.dim:
                    if d.HasField('dim_param'):
                        uses[d.dim_param] += 1
                        shapes[d.dim_param].add('%s %s' % (tag, v.name))
        print('=' * 70)
        print('%s  (%s)' % (label, os.path.basename(os.path.dirname(path))))
        for name, cnt in sorted(uses.items(), key=lambda kv: -kv[1]):
            ex = sorted(shapes[name])[:3]
            print('  %-28s appears %4d  e.g. %s' % (name, cnt, '; '.join(ex)))
        if not uses:
            print('  (no symbolic dims)')


if __name__ == '__main__':
    main()
