"""Audit every external-data region of a graph: does it fit the file, and does it collide with another?

The ERNIE crash is an access violation (0xC0000005, rc 139), not a caught std::bad_alloc -- the Qwen3
topology mutations die with bad_alloc at the same op. An access violation while lowering the FIRST
GroupQueryAttention is what you get when the tensor that node reads is memory-mapped from a region
that runs past the end of model.onnx.data, or that another tensor also claims. The chain of build
steps that grew the file (replicate_qkv_columns / tile_kv_heads for kv 2 -> 8, shrink_rope for the
cache rows) rewrote dims and lengths, but until now nothing ever checked the arithmetic of the
resulting offsets.

check_decl_conflicts.py compares an initializer's length against prod(dims)*itemsize; this script
compares the regions against the file and against each other. Reports:
  past EOF      offset + length > file size
  overlap       two regions whose [offset, offset+length) intervals intersect (fp16 caches and a
                shared-prefix layout are legitimate only when the bytes really are the same, so the
                overlap is printed with both tensor names for manual adjudication, not auto-failed)
  gap           unused bytes between consecutive regions (harmless, counted only)
  trailing      file bytes not claimed by any region

Usage: python -X utf8 check_data_regions.py e_v35 q3n0
"""
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
ITEM = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2, onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.INT64: 8, onnx.TensorProto.INT32: 4, onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1, onnx.TensorProto.BOOL: 1}


def audit(name):
    d = os.path.join(ROOT, name)
    m = onnx.load(os.path.join(d, 'model.onnx'), load_external_data=False)
    data = os.path.join(d, 'model.onnx.data')
    size = os.path.getsize(data) if os.path.exists(data) else 0
    regions = []
    for i in m.graph.initializer:
        ed = {e.key: e.value for e in i.external_data}
        if 'offset' not in ed:
            continue
        item = ITEM.get(i.data_type)
        n = 1
        for x in i.dims:
            n *= x
        need = n * item if item else 0
        regions.append((int(ed['offset']), int(ed['length']), i.name, need,
                        ','.join(str(x) for x in i.dims), item))
    regions.sort()
    print('== %s  file %d B  external initializers %d' % (name, size, len(regions)))
    past = [(o, l, nm) for o, l, nm, need, dims, item in regions if o + l > size]
    for o, l, nm in past:
        print('   PAST EOF   %-46s offset %d + length %d = %d > %d' % (nm, o, l, o + l, size))
    wrong = [(o, l, nm, need, dims, item) for o, l, nm, need, dims, item in regions
             if need and l != need]
    for o, l, nm, need, dims, item in wrong:
        print('   BAD LENGTH %-46s dims [%s] x %d B = %d, length says %d'
              % (nm, dims, item, need, l))
    overlaps = []
    for a, b in zip(regions, regions[1:]):
        if b[0] < a[0] + a[1]:
            overlaps.append((a, b))
    for a, b in overlaps[:12]:
        print('   OVERLAP    %s [%d,%d) vs %s [%d,%d)'
              % (a[2], a[0], a[0] + a[1], b[2], b[0], b[0] + b[1]))
    if len(overlaps) > 12:
        print('   ... and %d more overlapping pairs' % (len(overlaps) - 12))
    end = max((o + l for o, l, _, _, _, _ in regions), default=0)
    print('   regions %d; past EOF %d; length/dims mismatches %d; overlaps %d; '
          'last byte %d; unclaimed tail %d'
          % (len(regions), len(past), len(wrong), len(overlaps), end, size - end))
    return len(past), len(wrong), len(overlaps)


def main():
    names = sys.argv[1:] or ['e_v35']
    bad = 0
    for n in names:
        bad += sum(audit(n))
        print()
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
