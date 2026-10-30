"""Census a generated _ZG artifact and compare it against the board-proven Qwen3 reference.

The five-stage chain is only a pass if the generated network carries what the board driver expects, so
read the ZG json the same way for both builds and print the fields that the driver and the numeric check
depend on: op count, how many ops landed on @zhuget(330) versus @hostt (an op that fell to host is a
capability gap, not a failure), the network-input contract (count, shapes, memory types), params_bytes
and params_md5. Reference is Q3V1-decoder_ZG.json, whose params_md5 1081ddd7222dc2cadd5bee8dc27b781f is
the build already run on the board.

Usage: python -X utf8 zg_census.py <a_ZG.json> [<b_ZG.json> ...]
"""
import collections
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def census(path):
    d = json.load(io.open(path, encoding='utf-8'))
    ops = d['ops']
    o0 = [o for o in ops if o.get('op_id') == 0][0]
    outs = o0['outputs']
    consumed = {(v.get('v_id') if isinstance(v, dict) else None)
                for o in ops for v in (o.get('inputs') or [])}
    dead = [v['v_id'] for v in outs if v['v_id'] not in consumed]
    print('== %s' % path.replace('\\', '/'))
    for k in ('name', 'ai_target', 'framework_version', 'icraft_version', 'llm_mode',
              'params_bytes', 'params_md5', 'swap_mode'):
        print('   %-14s %s' % (k, d.get(k)))
    print('   ops            %d' % len(ops))
    print('   targets        %s' % collections.Counter(o.get('compile_target', '?') for o in ops).most_common())
    kinds = collections.Counter(o.get('_type_key', '').split('::')[-1] for o in ops)
    host_kinds = sorted({o.get('_type_key', '').split('::')[-1] for o in ops
                         if o.get('compile_target') == '@hostt'})
    print('   kinds          %s' % kinds.most_common(12))
    print('   host-only kinds %s' % host_kinds)
    print('   inputs         %d  (never consumed: %d %s)' % (len(outs), len(dead), dead[:8]))
    print('   in shapes      %s' % dict(collections.Counter(tuple(v['dtype']['shape']) for v in outs)))
    print('   in dtypes      %s' % collections.Counter(v['dtype']['element_dtype'] for v in outs).most_common())
    print('   in mtypes      %s' % collections.Counter(v['mtype'] for v in outs).most_common())
    print('   outputs        %d' % len([o for o in ops if o.get('_type_key', '').endswith('Output')]))
    on = [o for o in ops if o.get('_type_key', '').split('::')[-1] == 'Output']
    if on:
        ov = on[0].get('outputs') or []
        print('   out values     %d shapes %s' % (len(ov), dict(collections.Counter(tuple(v['dtype']['shape']) for v in ov))))
    raw = path.replace('.json', '.raw')
    if os.path.exists(raw):
        print('   raw            %d bytes  json==raw size? %s' % (os.path.getsize(raw), os.path.getsize(raw) == d['params_bytes']))
    return d


if __name__ == '__main__':
    for p in sys.argv[1:]:
        census(p)
