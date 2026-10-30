"""Shrink the declared rope-cache geometry of a graph, keeping the existing bytes as a prefix.

The split-qkv build no longer access-violates; it dies with std::bad_alloc while lowering the FIRST
GroupQueryAttention. The one number in that node that is far bigger than anything the Qwen3 graph
(which compiles) exposed is the rope cache: ERNIE declares cos_cache/sin_cache as [131072, 64]
(16,777,216 B each) while Qwen3 ships [40960, 64] and Qwen2-0.5B ships [32768, 32]. If icraft
materialises something proportional to the cache rows, 131072 is where it would blow up.

This is metadata-only: the external-data offsets stay, and length shrinks to rows*cols*2, so the
tensor is read as a prefix of the region that already holds it. cos sits at 509739008 and sin at
526516224 with 16,777,216 B each, and the next tensor (lm_head) starts at 543293440, so a shorter
length cannot collide with anything.

THE BUG THIS SCRIPT USED TO HAVE: the exporter also writes a value_info entry for cos_cache/sin_cache,
and the first version of this script edited only the initializer. Every build from e_v21 on therefore
declared a [131072, 64] rope cache while handing over bytes for [40960, 64], and the GroupQueryAttention
-- the only consumer of that cache -- access-violated every time, at the identical op, regardless of
head geometry, operand form, node order or attribute. check_decl_conflicts.py is the standing guard;
q3n0 (which parses) reports zero conflicts. Both declarations are now rewritten together.

Diagnostic first -- if this makes parse pass, the position cap becomes a documented compile-time
limit rather than the model's native 131072.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
SIZE = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.FLOAT: 4, onnx.TensorProto.BFLOAT16: 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v3')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--rows', type=int, default=40960)
    ap.add_argument('--name', action='append', default=[], help='initializer name, default both caches')
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, args.src)
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    targets = args.name or ['cos_cache', 'sin_cache']
    for tname in targets:
        init = [i for i in m.graph.initializer if i.name == tname]
        if not init:
            raise SystemExit('initializer %s not found' % tname)
        init = init[0]
        ed = {e.key: e.value for e in init.external_data}
        old_dims = list(init.dims)
        item = SIZE[init.data_type]
        cols = old_dims[-1]
        new_len = args.rows * cols * item
        print('%s: dims %s -> [%d, %d]; offset %s kept; length %s -> %d'
              % (tname, old_dims, args.rows, cols, ed['offset'], ed['length'], new_len))
        if new_len > int(ed['length']):
            raise SystemExit('refusing to grow the region: %d > %s' % (new_len, ed['length']))
        del init.dims[:]
        init.dims.extend([args.rows, cols])
        for e in init.external_data:
            if e.key == 'length':
                e.value = str(new_len)
        # and the declaration the exporter wrote for the same tensor, without which the parser reads
        # three times the bytes that are there
        decls = [v for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)
                 if v.name == tname]
        if not decls:
            print('%s: no value_info/graph-io declaration to sync' % tname)
        for v in decls:
            dd = v.type.tensor_type.shape.dim
            if len(dd) != len(old_dims):
                raise SystemExit('%s declared rank %d, initializer rank %d'
                                 % (tname, len(dd), len(old_dims)))
            for d in list(dd):
                if d.HasField('dim_param'):
                    raise SystemExit('%s is declared symbolic (%s); this script only bakes literals'
                                     % (tname, d.dim_param))
            del dd[:]
            for n in init.dims:
                dd.add().dim_value = n
            print('%s: %s declaration now %s'
                  % (tname, 'value_info' if v in list(m.graph.value_info) else 'graph io',
                   list(init.dims)))

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
    for tname in targets:
        i = [x for x in chk.graph.initializer if x.name == tname][0]
        ed = {e.key: e.value for e in i.external_data}
        dec = [[d.dim_value for d in v.type.tensor_type.shape.dim]
               for v in list(chk.graph.value_info) + list(chk.graph.input)
               if v.name == tname]
        print('after %s dims=%s offset=%s length=%s declared=%s'
              % (tname, list(i.dims), ed['offset'], ed['length'], dec))
    print('onnx', os.path.getsize(out), 'bytes; data', os.path.getsize(link), 'bytes')
    print('DONE_ROPE')


if __name__ == '__main__':
    main()
