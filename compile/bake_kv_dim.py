"""Materialise the symbolic head-dim axis of the ERNIE graph and copy everything else verbatim.

Why: icraft's own parse headers show the difference. Qwen3 (which compiled) prints
past_key_values.0.key;[1,8,0,128] -- four axes. ERNIE prints [1,2,0] -- three axes, because its
fourth axis is the dim_param "kv_cache_dim" and the shape resolver has no binding for that name, so
the axis is dropped. The access violation lands on the first GroupQueryAttention, the first op that
consumes those past tensors.

The value is not a guess: config.json says head_dim=128, and the qkv MatMul weight {1024 2560}
reconciles as 16*128 + 2*128 + 2*128 = 2560 (num_heads=16, kv_num_heads=2). Only the head dim is
baked -- sequence axes stay dynamic, because the toml's max_prompt_length/max_tokens_length drive
those.

The 966 MB external-data blob is hard-linked, not copied.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def dims_of(v):
    tt = v.type.tensor_type
    if not tt.HasField('shape'):
        return None
    return [d.dim_value if d.HasField('dim_value') else d.dim_param for d in tt.shape.dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v1')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--name', default='kv_cache_dim')
    ap.add_argument('--value', type=int, default=128)
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    before = {t: [x.name for x in getattr(g, t) if any(
        d.HasField('dim_param') and d.dim_param == args.name for d in x.type.tensor_type.shape.dim)]
        for t in ('input', 'output', 'value_info')}
    print('before: %s occurrences per collection:' % args.name,
          {k: len(v) for k, v in before.items()})

    touched = 0
    for t in ('input', 'output', 'value_info'):
        for v in getattr(g, t):
            tt = v.type.tensor_type
            if not tt.HasField('shape'):
                continue
            for d in tt.shape.dim:
                if d.HasField('dim_param') and d.dim_param == args.name:
                    d.ClearField('dim_param')
                    d.dim_value = args.value
                    touched += 1
    print('baked %d axes to %d' % (touched, args.value))
    if touched == 0:
        raise SystemExit('nothing baked -- symbolic name %r not found' % args.name)

    out = os.path.join(dst_dir, 'model.onnx')
    onnx.save(m, out)

    link = os.path.join(dst_dir, 'model.onnx.data')
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
            open(os.path.join(dst_dir, extra), 'w', encoding='utf-8', newline='\n').write(
                open(s, encoding='utf-8').read())

    chk = onnx.load(out, load_external_data=False)
    left = set()
    for t in ('input', 'output', 'value_info'):
        for v in getattr(chk.graph, t):
            tt = v.type.tensor_type
            if not tt.HasField('shape'):
                continue
            for d in tt.shape.dim:
                if d.HasField('dim_param'):
                    left.add(d.dim_param)
    print('after: remaining symbolic names', sorted(left))
    for nm in ('past_key_values.0.key', 'present.0.key', 'past_key_values.17.value'):
        for v in list(chk.graph.input) + list(chk.graph.output):
            if v.name == nm:
                print('  %-26s %s' % (nm, dims_of(v)))
    print('onnx', os.path.getsize(out), 'bytes; data', os.path.getsize(link), 'bytes')
    print('DONE_BAKE')


if __name__ == '__main__':
    main()
