"""Control for the crop experiment: re-serialize a parsing graph without changing its structure.

Q3H (q3g0 cropped to 18 layers) dies exactly like every ERNIE build -- 2 network_input tensors, the
same 93,697-byte log, access violation entering the first GroupQueryAttention. But between q3g0 and
q3h0 two things changed at once: the number of layers AND the fact that the file was rewritten by
onnx.save with load_external_data=False. If that rewrite alone broke cache registration, the whole
layer-count conclusion is void.

So copy q3g0 through this script: load the protobuf with no external data, save it unchanged, hardlink
the .data. If the round-tripped graph parses (58 network tensors), my serialization is innocent and a
crop at 27 layers failing would really mean layer count. If it also dies, the crop results are
uninterpretable and the fix belongs in how the graph is written out.

Usage: python -X utf8 roundtrip_control.py --src q3g0 --dst q3g0_rt
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
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)
    m = onnx.load(os.path.join(src, 'model.onnx'), load_external_data=False)
    g = m.graph
    before = (len(g.node), len(g.initializer), len(g.input), len(g.output), len(g.value_info))
    onnx.save(m, os.path.join(dst, 'model.onnx'))
    s = os.path.join(src, 'model.onnx.data')
    t = os.path.join(dst, 'model.onnx.data')
    if os.path.exists(s):
        if os.path.exists(t):
            os.remove(t)
        os.link(s, t)
    chk = onnx.load(os.path.join(dst, 'model.onnx'), load_external_data=False)
    cg = chk.graph
    after = (len(cg.node), len(cg.initializer), len(cg.input), len(cg.output), len(cg.value_info))
    print('%s -> %s' % (args.src, args.dst))
    print('nodes/initializer/input/output/value_info  before=%s  after=%s  identical=%s'
          % (before, after, before == after))
    print('onnx bytes %d -> %d ; data %d (same inode: %s)'
          % (os.path.getsize(os.path.join(src, 'model.onnx')),
             os.path.getsize(os.path.join(dst, 'model.onnx')),
             os.path.getsize(t) if os.path.exists(t) else -1,
             os.path.exists(t) and os.path.samefile(s, t)))
    print('DONE_ROUNDTRIP')


if __name__ == '__main__':
    main()
