"""Dump the layer-0 GroupQueryAttention signature of a graph: attrs, operand kinds, dtypes, shapes.

Used to diff the ERNIE node against the Qwen3 node that parses (q3n0), field by field, because the
geometry now matches (Q 2048 / K 1024 / V 1024, 16 heads / 8 kv heads / head_dim 128) and the crash
point is identical in both producer topologies.
"""
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')


def main():
    for d in sys.argv[1:]:
        m = onnx.load(os.path.join(ROOT, d, 'model.onnx'), load_external_data=False)
        g = m.graph
        bo = {}
        for n in g.node:
            for o in n.output:
                if o:
                    bo[o] = n
        inits = {i.name: i for i in g.initializer}
        vis = {}
        for v in list(g.input) + list(g.output) + list(g.value_info):
            vis[v.name] = v
        print('=== %s: ir_version %d opset %s nodes %d inputs %d outputs %d producer %s'
              % (d, m.ir_version, [(o.domain, o.version) for o in m.opset_import],
                 len(g.node), len(g.input), len(g.output),
                 (m.producer_name, m.producer_version)))
        node = [n for n in g.node if n.op_type == 'GroupQueryAttention'][0]
        print('    node %s outputs %s' % (node.name, list(node.output)))
        print('    attrs:')
        for a in node.attribute:
            print('      %-22s type=%d i=%s f=%s s=%s' % (a.name, a.type, a.i, a.f,
                                                          a.s.decode() if a.s else ''))
        print('    inputs:')
        for i, ip in enumerate(node.input):
            if ip == '':
                print('      [%2d] <empty>' % i)
                continue
            kind = 'INIT' if ip in inits else ('NODE:%s' % bo[ip].op_type if ip in bo
                                               else 'GRAPH_INPUT' if any(x.name == ip
                                                                         for x in g.input)
                                               else '??')
            dt = shape = ''
            if ip in inits:
                t = inits[ip]
                dt = onnx.TensorProto.DataType.Name(t.data_type)
                shape = str(list(t.dims))
            elif ip in vis:
                vi = vis[ip]
                dt = onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
                shape = str([str(x.dim_value if x.HasField('dim_value') else x.dim_param)
                             for x in vi.type.tensor_type.shape.dim])
            print('      [%2d] %-58s %-12s %-8s %s' % (i, ip, kind, dt, shape))
        print('    outputs shapes:')
        for o in node.output:
            vi = vis.get(o)
            print('      %-58s %s %s' % (o, onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
                                         if vi is not None else '?',
                                         str([str(x.dim_value if x.HasField('dim_value') else x.dim_param)
                                              for x in vi.type.tensor_type.shape.dim])
                                         if vi is not None else 'no shape'))
        for nm in ('cos_cache', 'sin_cache'):
            if nm in inits:
                t = inits[nm]
                print('    %s dtype %s dims %s ext %s'
                      % (nm, onnx.TensorProto.DataType.Name(t.data_type), list(t.dims),
                         t.data_location))
        # the mask-producing chain, if any
        for i, ip in enumerate(node.input):
            p = bo.get(ip)
            if p is not None and 'mask' in ip:
                chain = []
                cur = p
                while cur is not None and len(chain) < 8:
                    chain.append('%s(%s)' % (cur.op_type, cur.name))
                    nxt = [bo.get(x) for x in cur.input if x in bo]
                    cur = nxt[0] if nxt else None
                print('    mask chain slot %d: %s' % (i, ' <- '.join(chain)))


if __name__ == '__main__':
    main()
