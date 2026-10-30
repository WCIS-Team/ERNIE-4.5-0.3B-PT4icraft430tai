"""Clone e_v1's model.onnx with GroupQueryAttention attribute overrides; .data is hard-linked.

Diagnostic only — flipping rotary_interleaved changes rope semantics, so a successful parse here
does NOT mean the graph is correct; it only localises icraft-parser's AV.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v1')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--set', action='append', default=[],
                    help='attr=value applied to every GroupQueryAttention, e.g. rotary_interleaved=0')
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    pairs = []
    for kv in args.set:
        k, v = kv.split('=', 1)
        pairs.append((k, int(v) if v.lstrip('-').isdigit() else v))

    touched = 0
    for n in m.graph.node:
        if n.op_type != 'GroupQueryAttention':
            continue
        for k, v in pairs:
            hit = [a for a in n.attribute if a.name == k]
            if not hit:
                print('MISSING attr %s on %s' % (k, n.name))
                continue
            a = hit[0]
            old = a.i if a.type == a.INT else (a.f if a.type == a.FLOAT else a.s)
            if a.type == a.INT:
                a.i = int(v)
            elif a.type == a.FLOAT:
                a.f = float(v)
            else:
                a.s = str(v).encode()
            print('%s %s: %s -> %s' % (n.name, k, old, v))
        touched += 1
    print('gqa nodes touched=%d' % touched)

    out = os.path.join(dst, 'model.onnx')
    onnx.save(m, out)
    link = os.path.join(dst, 'model.onnx.data')
    if not os.path.exists(link):
        try:
            os.link(os.path.join(src_dir, 'model.onnx.data'), link)
            print('hardlinked data')
        except OSError as e:
            print('hardlink failed (%s), copying' % e)
            import shutil
            shutil.copyfile(os.path.join(src_dir, 'model.onnx.data'), link)
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst, extra), 'w', encoding='utf-8', newline='\n').write(
                open(s, encoding='utf-8').read())
    print('out', out, os.path.getsize(out), 'bytes')
    print('DONE_ATTR')


if __name__ == '__main__':
    main()
