"""Probe the ERNIE genai graph for the pieces the compile chain consumes:
Transpose presence, embedding/lm_head initializers, full IO contract, rope caches, mask dtype."""
import collections
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P = sys.argv[1] if len(sys.argv) > 1 else r'S:\apps\llm\agentwork\ernie\compile\fmodel\model.onnx'
m = onnx.load(P, load_external_data=False)
g = m.graph
DT = onnx.TensorProto.DataType.Name


def dims(vi):
    t = vi.type.tensor_type
    return '%s%s' % (DT(t.elem_type), [d.dim_value if d.HasField('dim_value') else d.dim_param
                                       for d in t.shape.dim])


print('ir=%d opset=%s nodes=%d inits=%d in=%d out=%d' % (
    m.ir_version, [(o.domain or 'ai.onnx', o.version) for o in m.opset_import],
    len(g.node), len(g.initializer), len(g.input), len(g.output)))

print('--- Transpose nodes (the detranspose step only makes sense if these exist) ---')
tr = [n for n in g.node if n.op_type == 'Transpose']
print('count=%d' % len(tr))
for n in tr:
    print('  %-52s in=%s out=%s perm=%s' % (
        n.name, list(n.input), list(n.output),
        [onnx.helper.get_attribute_value(a) for a in n.attribute]))

print('--- nodes whose name mentions embed / lm_head / logits ---')
for n in g.node:
    if any(k in n.name.lower() for k in ('embed', 'lm_head', 'output_layer', 'logit')):
        print('  %-22s %-46s in=%s out=%s' % (n.op_type, n.name, list(n.input)[:3], list(n.output)))

print('--- initializers: embedding / lm_head / rope ---')
for i in g.initializer:
    if any(k in i.name.lower() for k in ('embed', 'lm_head', 'cos', 'sin', 'cache')):
        ed = {e.key: e.value for e in i.external_data}
        print('  %-46s %s %s ext=%s' % (i.name, DT(i.data_type), list(i.dims),
                                        {k: ed[k] for k in ('offset', 'length') if k in ed}))

print('--- FULL input contract (%d) ---' % len(g.input))
for i, vi in enumerate(g.input):
    print('  [%2d] %-34s %s' % (i, vi.name, dims(vi)))
print('--- FULL output contract (%d) ---' % len(g.output))
for i, vi in enumerate(g.output):
    print('  [%2d] %-34s %s' % (i, vi.name, dims(vi)))

print('--- op histogram ---')
h = collections.Counter(n.op_type for n in g.node)
for k, v in h.most_common():
    print('  %-28s %d' % (k, v))

print('--- first 12 nodes (embedding entry region) ---')
for n in list(g.node)[:12]:
    print('  %-22s %-46s in=%d out=%s' % (n.op_type, n.name, len(n.input), list(n.output)))
