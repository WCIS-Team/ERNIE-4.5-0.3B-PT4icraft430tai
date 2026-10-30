"""One-off: disk headroom plus the exact geometry a qkv-column-replication rewrite needs."""
import shutil
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for drive in ('S:\\', 'C:\\'):
    u = shutil.disk_usage(drive)
    print('%s free %.1f GB / total %.1f GB' % (drive, u.free / 2 ** 30, u.total / 2 ** 30))

g = onnx.load('S:/apps/llm/agentwork/ernie/compile/fmodel/e_v14/model.onnx',
              load_external_data=False).graph
inits = {i.name: i for i in g.initializer}
for nm in ('model.layers.0.attn.qkv_proj.MatMul.weight',
           'model.layers.0.attn.o_proj.MatMul.weight',
           '/model/constants/INT64/[2048, 256, 256]',
           'cos_cache'):
    i = inits.get(nm)
    if i is None:
        print('%-52s MISSING' % nm)
        continue
    print('%-52s dims=%s dt=%s loc=%s ext=%s'
          % (nm, list(i.dims), i.data_type, i.data_location,
             {e.key: str(e.value) for e in i.external_data}))
tot = sum(int(str(e.value)) for i in g.initializer for e in i.external_data if e.key == 'length')
print('sum of external lengths %d bytes; file %d'
      % (tot, __import__('os').path.getsize(
          'S:/apps/llm/agentwork/ernie/compile/fmodel/e_v14/model.onnx.data')))
print('qkv weights across layers:',
      sorted({tuple(list(x.dims)) for k, x in inits.items() if 'qkv_proj' in k}))
biases = [k for k in inits if 'qkv_proj' in k and 'weight' not in k]
print('qkv-side non-weight initializers:', biases[:6], len(biases))
