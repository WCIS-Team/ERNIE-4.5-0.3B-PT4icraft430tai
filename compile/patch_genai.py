"""Variant dir whose genai_config.json declares a shorter context, everything else byte-identical.

Tests whether icraft's LLM front end reads context_length/max_length from genai_config.json and
sizes something from it. ERNIE ships 131072 where the Qwen3 graph that compiles ships 40960, and the
shrink of the rope-cache initializers alone did not move the bad_alloc, so the config is the other
half of that number.

model.onnx and the .data are hard-linked so the variant costs nothing on disk.
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--set', action='append', default=[], help='dotted path=value, e.g. model.context_length=40960')
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    dst = os.path.join(ROOT, args.dst)
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(os.path.join(src, 'genai_config.json'), encoding='utf-8'))
    for kv in args.set:
        path, val = kv.split('=', 1)
        cur = cfg
        parts = path.split('.')
        for p in parts[:-1]:
            if p not in cur:
                raise SystemExit('key %s not present in genai_config.json' % path)
            cur = cur[p]
        old = cur.get(parts[-1], '<absent>')
        cur[parts[-1]] = int(val) if val.lstrip('-').isdigit() else val
        print('%s: %s -> %s' % (path, old, cur[parts[-1]]))

    with open(os.path.join(dst, 'genai_config.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    for name in ('config.json',):
        s = os.path.join(src, name)
        if os.path.exists(s):
            open(os.path.join(dst, name), 'w', encoding='utf-8', newline='\n').write(open(s, encoding='utf-8').read())
    for name in ('model.onnx', 'model.onnx.data'):
        d = os.path.join(dst, name)
        if not os.path.exists(d):
            os.link(os.path.join(src, name), d)
    print('linked onnx', os.path.getsize(os.path.join(dst, 'model.onnx')),
          'bytes; data', os.path.getsize(os.path.join(dst, 'model.onnx.data')), 'bytes')
    print('DONE_GENAI')


if __name__ == '__main__':
    main()
