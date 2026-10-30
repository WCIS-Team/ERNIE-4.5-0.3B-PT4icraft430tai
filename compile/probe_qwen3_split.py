"""Dump the qkv_proj/Split node of the working Qwen3 graph: attributes and input forms.

The split-QKV rewrite for ERNIE has to use a form icraft's parser demonstrably accepts, so copy the
proven node definition instead of inventing one (opset 13+ moved Split sizes from attribute to
input, and only one of those forms is guaranteed to have been exercised here).
"""
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'qwen3', 'compile',
                 'fmodel', 'q3i_v1', 'model.onnx')

m = onnx.load(P, load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}


def dt(t):
    return onnx.TensorProto.DataType.Name(t)


for n in g.node:
    if 'qkv_proj' not in n.name or n.op_type not in ('Split', 'Slice', 'MatMul', 'Add', 'Reshape'):
        continue
    print('%-8s %s' % (n.op_type, n.name))
    print('   attrs:', [(a.name, onnx.helper.get_attribute_value(a)) for a in n.attribute])
    for k, i in enumerate(n.input):
        if not i:
            print('   in[%d] <EMPTY>' % k)
            continue
        it = inits.get(i)
        if it is not None:
            import numpy as np
            arr = onnx.numpy_helper.to_array(it) if it.data_location != onnx.TensorProto.EXTERNAL else None
            print('   in[%d] %-45s INIT %s %s %s' % (k, i, dt(it.data_type), list(it.dims),
                                                     None if arr is None else arr.reshape(-1)[:8].tolist()))
        else:
            print('   in[%d] %s' % (k, i))
    print('   outs:', list(n.output))
print('opset', [(o.domain or 'ai.onnx', o.version) for o in m.opset_import])
