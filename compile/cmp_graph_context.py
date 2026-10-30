"""Last structural comparison left: what differs between the two graphs OUTSIDE the attention block.

The layer-0 GroupQueryAttention of e_v8/e_v11 now matches q3i_v1's node for node -- same 7 attributes,
same 12 input slots with the same dtypes/shapes/producers' output shapes, same 3 outputs, same mask
reformat subgraph, same rope cache dtype -- and yet icraft dies entering it (op_id stops at 8) while
q3i_v1 lowers it from op_id 12 onward. So if the cause is structural at all, it is somewhere else in
the graph and the parser reaches it while building the attention expansion. Print every whole-graph
difference: op-type census, graph input contract, graph output contract, opset/ir metadata, and the
initializer dtype/size census.
"""
import os
import sys
from collections import Counter

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

A = sys.argv[1] if len(sys.argv) > 2 else os.path.join(
    'S:', os.sep, 'apps', 'llm', 'agentwork', 'qwen3', 'compile', 'fmodel', 'q3i_v1', 'model.onnx')
B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    'S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel', 'e_v11', 'model.onnx')


def label(p):
    return os.path.basename(os.path.dirname(p))


def dims(v):
    return [d.dim_value if d.HasField('dim_value') else d.dim_param
            for d in v.type.tensor_type.shape.dim]


def load(p):
    return onnx.load(p, load_external_data=False)


def main():
    for p in (A, B):
        if not os.path.exists(p):
            raise SystemExit('missing %s' % p)
    ma, mb = load(A), load(B)
    ga, gb = ma.graph, mb.graph
    print('%-22s | %-22s | %s' % (label(A), label(B), 'field'))
    for f, va, vb in (('ir_version', ma.ir_version, mb.ir_version),
                      ('producer_name', ma.producer_name, mb.producer_name),
                      ('model_version', ma.model_version, mb.model_version),
                      ('opset_import', sorted((o.domain or 'ai.onnx', o.version) for o in ma.opset_import),
                       sorted((o.domain or 'ai.onnx', o.version) for o in mb.opset_import)),
                      ('graph.name', ga.name, gb.name),
                      ('metadata_props', [(k.p, str(k.s)[:60]) for k in ma.metadata_props],
                       [(k.p, str(k.s)[:60]) for k in mb.metadata_props]),
                      ('nodes', len(ga.node), len(gb.node)),
                      ('initializers', len(ga.initializer), len(gb.initializer)),
                      ('value_info', len(ga.value_info), len(gb.value_info))):
        print('%-22s | %-22s | %s' % (str(va)[:22], str(vb)[:22], f))

    print()
    print('== op-type census (only where the counts differ) ==')
    ca, cb = Counter(n.op_type for n in ga.node), Counter(n.op_type for n in gb.node)
    for op in sorted(set(ca) | set(cb)):
        if ca.get(op, 0) != cb.get(op, 0):
            print('  %-32s %4d vs %4d' % (op, ca.get(op, 0), cb.get(op, 0)))
    print('  identical op types: %s' % sorted(op for op in set(ca) & set(cb)
                                             if ca[op] == cb[op]))

    print()
    print('== graph input contract ==')
    na = {v.name: v for v in ga.input}
    nb = {v.name: v for v in gb.input}
    for name in sorted(set(na) | set(nb)):
        sa = '%s %s' % (onnx.TensorProto.DataType.Name(na[name].type.tensor_type.elem_type),
                        dims(na[name])) if name in na else '-- ABSENT'
        sb = '%s %s' % (onnx.TensorProto.DataType.Name(nb[name].type.tensor_type.elem_type),
                        dims(nb[name])) if name in nb else '-- ABSENT'
        if str(sa) != str(sb) or name not in na or name not in nb:
            print('  %-28s %-34s | %-34s' % (name[:28], sa, sb))

    print()
    print('== graph output contract ==')
    oa = {v.name: v for v in ga.output}
    ob = {v.name: v for v in gb.output}
    for name in sorted(set(oa) | set(ob)):
        sa = '%s %s' % (onnx.TensorProto.DataType.Name(oa[name].type.tensor_type.elem_type),
                        dims(oa[name])) if name in oa else '-- ABSENT'
        sb = '%s %s' % (onnx.TensorProto.DataType.Name(ob[name].type.tensor_type.elem_type),
                        dims(ob[name])) if name in ob else '-- ABSENT'
        if str(sa) != str(sb) or name not in oa or name not in ob:
            print('  %-28s %-34s | %-34s' % (name[:28], sa, sb))

    print()
    print('== initializer dtype census ==')
    ia = Counter(onnx.TensorProto.DataType.Name(i.data_type) for i in ga.initializer)
    ib = Counter(onnx.TensorProto.DataType.Name(i.data_type) for i in gb.initializer)
    print('  %s | %s' % (dict(ia), dict(ib)))
    big_a = sorted(((i.name, list(i.dims)) for i in ga.initializer), key=lambda x: -max(x[1][0], 1))[0]
    big_b = sorted(((i.name, list(i.dims)) for i in gb.initializer), key=lambda x: -max(x[1][0], 1))[0]
    print('  widest initializer: %s %s | %s %s' % (label(A), big_a, label(B), big_b))

    print()
    print('== first 14 nodes in graph order ==')
    for i in range(14):
        print('  %2d %-26s | %2d %-26s' % (
            i, ga.node[i].op_type + ':' + ga.node[i].name[:34],
            i, gb.node[i].op_type + ':' + gb.node[i].name[:34]))


if __name__ == '__main__':
    main()
