"""Metadata-only variant of the ERNIE genai ONNX so icraft-parse can load it.

Same recipe as the Qwen3 line: ir 10 -> 8, ai.onnx opset 22 -> 17, and move the one
ai.onnx-domain SimplifiedLayerNormalization into com.microsoft (the 36
SkipSimplifiedLayerNormalization already live there). The .data blob is hard-linked, not copied.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
SRC = os.path.join(ROOT, 'e_exported')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dst', required=True)
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--ir', type=int, default=0)
    ap.add_argument('--opset', type=int, default=0)
    ap.add_argument('--sln-ms', action='store_true')
    args = ap.parse_args()

    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)

    m = onnx.load(os.path.join(args.src, 'model.onnx'), load_external_data=False)
    print('before ir=%d opset=%s' % (m.ir_version, [(o.domain or 'ai.onnx', o.version) for o in m.opset_import]))

    if args.ir:
        m.ir_version = args.ir
    if args.opset:
        for oi in m.opset_import:
            if oi.domain in ('', 'ai.onnx'):
                oi.version = args.opset
    moved = 0
    if args.sln_ms:
        for n in m.graph.node:
            if n.op_type == 'SimplifiedLayerNormalization' and (n.domain or '') == '':
                n.domain = 'com.microsoft'
                moved += 1
    print('after ir=%d opset=%s sln_moved=%d'
          % (m.ir_version, [(o.domain or 'ai.onnx', o.version) for o in m.opset_import], moved))

    out = os.path.join(dst, 'model.onnx')
    onnx.save(m, out)

    data_link = os.path.join(dst, 'model.onnx.data')
    if not os.path.exists(data_link):
        try:
            os.link(os.path.join(args.src, 'model.onnx.data'), data_link)
            print('hardlinked data', data_link)
        except OSError as e:
            print('hardlink failed (%s), copying' % e)
            import shutil
            shutil.copyfile(os.path.join(args.src, 'model.onnx.data'), data_link)
    for extra in ('genai_config.json',):
        s = os.path.join(args.src, extra)
        if os.path.exists(s):
            open(os.path.join(dst, extra), 'w', encoding='utf-8', newline='\n').write(
                open(s, encoding='utf-8').read())

    print('out', out, os.path.getsize(out), 'bytes; data', os.path.getsize(data_link), 'bytes')
    print('DONE_META')


if __name__ == '__main__':
    main()
