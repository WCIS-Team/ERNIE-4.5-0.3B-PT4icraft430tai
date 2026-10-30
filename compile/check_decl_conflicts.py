"""Report every tensor whose initializer dims disagree with its value_info dims, plus the external bytes.

Why this exists: shrink_rope.py cropped cos_cache/sin_cache by rewriting the *initializer* dims and the
external-data length, but the exporter also writes a value_info entry for those same tensors, and it was
left at the original [131072, 64]. A graph that declares 131072 rows and hands over bytes for 40960 rows
is one out-of-bounds read waiting to happen, and the only node that consumes the rope cache is the
GroupQueryAttention -- which is exactly where every ERNIE parse dies. This is the general check for that
class of bug: any script that edits one declaration and not the other.

Only reads; prints and exits non-zero when a contradiction is found.
Usage: python -X utf8 check_decl_conflicts.py e_v27 [q3n0 ...]
"""
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
ITEM = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2, onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.INT64: 8, onnx.TensorProto.INT32: 4}


def dims_of(tt):
    return [(d.dim_value if d.HasField('dim_value') else d.dim_param)
            for d in tt.shape.dim]


def main():
    total = 0
    for name in sys.argv[1:]:
        g = onnx.load(os.path.join(ROOT, name, 'model.onnx'), load_external_data=False).graph
        inits = {}
        for i in g.initializer:
            inits.setdefault(i.name, []).append(i)
        vis = {}
        for v in list(g.value_info) + list(g.input) + list(g.output):
            vis.setdefault(v.name, []).append(v)
        print('== %s  initializers %d  declared tensors %d'
              % (name, len(g.initializer), len(g.value_info)))
        dup = {k: len(v) for k, v in inits.items() if len(v) > 1}
        if dup:
            print('   DUPLICATE initializer names: %s' % dup)
        bad = 0
        for k, iv in sorted(inits.items()):
            if k not in vis:
                continue
            want = list(iv[0].dims)
            for v in vis[k]:
                got = dims_of(v.type.tensor_type)
                if got != want:
                    ed = {e.key: e.value for e in iv[0].external_data}
                    nbytes = None
                    if 'length' in ed:
                        nbytes = int(ed['length'])
                        declared = 1
                        for d in got:
                            declared *= d if isinstance(d, int) else 0
                        declared *= ITEM.get(iv[0].data_type, 0)
                    print('   CONFLICT %s: initializer %s vs %s %s; bytes %s, declared would need %s'
                          % (k, want,
                             'value_info' if v in list(g.value_info) else 'graph io',
                             got, nbytes, declared if nbytes else '-'))
                    bad += 1
        total += bad
        # external-data length vs its own dims, for every initializer
        short = 0
        for k, iv in inits.items():
            i = iv[0]
            ed = {e.key: e.value for e in i.external_data}
            if 'length' not in ed:
                continue
            want = 1
            for d in i.dims:
                want *= d
            want *= ITEM.get(i.data_type, 0)
            if int(ed['length']) != want:
                short += 1
                if short <= 5:
                    print('   LENGTH %s: dims %s need %d B, external length %s'
                          % (k, list(i.dims), want, ed['length']))
        print('   value_info conflicts %d; initializer length/dims mismatches %d' % (bad, short))
    if total:
        raise SystemExit('%d declaration conflicts' % total)


if __name__ == '__main__':
    main()
