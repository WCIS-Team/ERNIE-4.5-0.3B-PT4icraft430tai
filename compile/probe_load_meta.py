"""Why does onnxruntime refuse the rewritten graph but not the export it came from?

Print each model's opset_import, its ir_version, and the producer, then try InferenceSession on each and
report the exception text. The rewritten e_v35 fails to load with
  Fatal error: com.microsoft:SimplifiedLayerNormalization(-1) is not a registered function/op
while the export this pipeline was built from loaded fine in verify_onnx_ernie.py -- so either the
metadata rewrite (meta_variant.py) changed something ORT keys its custom-op registry on, or the SLN nodes
themselves were touched. This tells which, without guessing.
"""
import os
import sys

import onnx
import onnxruntime as ort
import onnxruntime_genai  # noqa: F401

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for d in sys.argv[1:]:
    p = os.path.join(d, 'model.onnx')
    m = onnx.load(p, load_external_data=False)
    print('== %s ir=%s producer=%s opset=%s'
          % (d, m.ir_version, m.producer_name,
             [(o.domain, o.version) for o in m.opset_import]), flush=True)
    doms = {}
    for n in m.graph.node:
        if n.domain:
            doms.setdefault(n.domain, set()).add(n.op_type)
    print('   node domains %s' % {k: sorted(v) for k, v in doms.items()}, flush=True)
    print('   nodes with empty domain that are not standard-17: %s'
          % sorted({n.op_type for n in m.graph.node
                    if not n.domain and n.op_type in
                    ('SimplifiedLayerNormalization', 'GroupQueryAttention', 'SkipSimplifiedLayerNormalization')}),
          flush=True)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
        print('   LOAD ok', flush=True)
    except Exception as e:
        print('   LOAD FAIL %s' % str(e)[:300], flush=True)
