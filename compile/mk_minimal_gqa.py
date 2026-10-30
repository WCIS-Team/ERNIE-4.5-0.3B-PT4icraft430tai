"""Build one-node LLM graphs so icraft's GroupQueryAttention lowering can be tested without context.

Where the bisect stands. e_v14's layer-0 GQA now agrees with q3i_v1's -- which compiles -- on all
seven attributes, all twelve input slots (dtypes, shapes, which slots are empty), the three outputs,
the rope cache dtype and row count, the mask-reformat contract feeding slots 5 and 6, the graph input
contract as the parser prints it (input_ids;[1,0] attention_mask;[1,32768] past;[1,8,0,128]), ir 8,
opset ('',17)+('com.microsoft',1) and producer onnxruntime-genai. Each of these was changed and
re-parsed on the real graph one at a time:

  kv head geometry 2 -> 8 (K,V 256 -> 1024 wide)      still dies entering the node
  rope rows 131072 -> 40960                           still dies
  rotary_interleaved 1 -> 0                           still dies
  do_rotary 1 -> 0 (and both together)                still dies
  sizes operand initializer -> Constant node          still dies
  identity Reshape on Q,K,V (54 nodes)                still dies
  Q,K,V handed in as graph inputs                     still dies

The parse log stops at op_id 8 in every one of those runs -- the node never prints a line. So either
this configuration genuinely cannot be lowered and the graph-level evidence is misleading me, or the
parser needs something that only exists *beside* the node. A single-node graph answers which, because
it has no beside: if the ERNIE-shaped node fails there while the Qwen3-shaped node passes, the cause
is in the node's own numbers and the remaining suspects are the ones I cannot see in an attribute
dump (the rope cache contents, the head_dim/2 slice arithmetic at head_dim 128). If both fail, the
harness is missing the LLM contract and nothing is concluded. If both pass, the node is fine and the
next step is deleting the attention from e_v14 to see whether the rest of the graph parses.

Variants:
  ernie        num_heads 16, kv_num_heads 8, K/V 1024, Q 2048, rope rows 40960, interleaved 1
  q3attrs      identical except rotary_interleaved 0 (the value the compiling graph uses)
  native       ERNIE's real geometry: kv_num_heads 2, K/V 256, Q 2048, rope rows 131072, interleaved 1
"""
import os
import sys

import numpy as np
import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile')
FM = os.path.join(ROOT, 'fmodel')
HD, NH, SEQ = 128, 16, 1024

VARIANTS = {
    'ernie': dict(knh=8, rows=40960, ri=1),
    'q3attrs': dict(knh=8, rows=40960, ri=0),
    'native': dict(knh=2, rows=131072, ri=1),
}


def vi(name, dims, dt=onnx.TensorProto.FLOAT16):
    return onnx.helper.make_tensor_value_info(name, dt, dims)


TEMPLATE = """[parse]
net_name = 'GQA@U@'
framework = 'onnx'
network = './fmodel/gqa_@T@/model.onnx'
jr_path = './jr_path/GQA@U@/'
max_prompt_length = 32
max_tokens_length = 512
target = 'zhuge'
llm_mode = true
quantized = true

[optimize]
target = 'zhuge'
json = './jr_path/GQA@U@//GQA@U@-decoder_parsed.json'
raw = './jr_path/GQA@U@//GQA@U@-decoder_parsed.raw'
jr_path = './jr_path/GQA@U@/'

[quantize]
json = './jr_path/GQA@U@//GQA@U@-decoder_optimized.json'
raw = './jr_path/GQA@U@//GQA@U@-decoder_optimized.raw'
jr_path = './jr_path/GQA@U@/'
qdtype = "fp16"
target = "zhuge"

[adapt]
target = 'zhuge'
json = './jr_path/GQA@U@//GQA@U@-decoder_quantized.json'
raw = './jr_path/GQA@U@//GQA@U@-decoder_quantized.raw'
jr_path = './jr_path/GQA@U@/'
pass_off = "icraft.MergeTransDataOpSeqPass"

[generate]
json = './jr_path/GQA@U@//GQA@U@-decoder_adapted.json'
raw = './jr_path/GQA@U@//GQA@U@-decoder_adapted.raw'
jr_path = './jr_path/GQA@U@/'
"""


def build(tag, knh, rows, ri):
    kvw = knh * HD
    q = vi('q', ['batch_size', 'sequence_length', NH * HD])
    k = vi('k', ['batch_size', 'sequence_length', kvw])
    v = vi('v', ['batch_size', 'sequence_length', kvw])
    pk = vi('past_key_values.0.key', ['batch_size', knh, 'past_sequence_length', HD])
    pv = vi('past_key_values.0.value', ['batch_size', knh, 'past_sequence_length', HD])
    mlen = vi('attn_mask_len', ['batch_size'], onnx.TensorProto.INT32)
    mstart = vi('attn_mask_start', [], onnx.TensorProto.INT32)

    rng = np.random.default_rng(0)
    cos = (rng.random((rows, HD // 2), dtype=np.float32) * 2 - 1).astype(np.float16)
    sin = (rng.random((rows, HD // 2), dtype=np.float32) * 2 - 1).astype(np.float16)
    ci = onnx.numpy_helper.from_array(cos, name='cos_cache')
    si = onnx.numpy_helper.from_array(sin, name='sin_cache')

    gqa = onnx.helper.make_node(
        'GroupQueryAttention',
        ['q', 'k', 'v', 'past_key_values.0.key', 'past_key_values.0.value',
         'attn_mask_len', 'attn_mask_start', 'cos_cache', 'sin_cache', '', '', ''],
        ['/model/layers.0/attn/GroupQueryAttention/output_0', 'present.0.key', 'present.0.value'],
        name='/model/layers.0/attn/GroupQueryAttention',
        domain='com.microsoft',
        num_heads=NH, kv_num_heads=knh, do_rotary=1, rotary_interleaved=ri,
        scale=0.0883883461356163, softcap=0.0, local_window_size=-1)

    graph = onnx.helper.make_graph(
        [gqa], 'main_graph',
        [q, k, v, pk, pv, mlen, mstart],
        [vi('/model/layers.0/attn/GroupQueryAttention/output_0',
            ['batch_size', 'sequence_length', NH * HD]),
         vi('present.0.key', ['batch_size', knh, 'total_sequence_length', HD]),
         vi('present.0.value', ['batch_size', knh, 'total_sequence_length', HD])],
        initializer=[ci, si])
    m = onnx.helper.make_model(graph, producer_name='onnxruntime-genai')
    m.ir_version = 8
    m.opset_import[0].version = 17
    m.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))
    try:
        onnx.checker.check_model(m, check_custom_domain=True)
        print('%-8s checker: ok' % tag)
    except Exception as e:
        print('%-8s checker says: %s' % (tag, str(e)[:200]))

    d = os.path.join(FM, 'gqa_' + tag)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'model.onnx')
    onnx.save(m, p)

    U = tag.upper()
    toml = TEMPLATE.replace('@U@', U).replace('@T@', tag)
    tpath = os.path.join(ROOT, 'GQA%s_parse.toml' % tag.upper())
    open(tpath, 'w', encoding='utf-8', newline='\n').write(toml)
    print('%-8s knh=%d kvw=%d rows=%d ri=%d onnx=%d B  toml=%s'
          % (tag, knh, kvw, rows, ri, os.path.getsize(p), os.path.basename(tpath)))
    return p


def main():
    for tag, kw in VARIANTS.items():
        build(tag, **kw)
    print('DONE_MINIMAL')


if __name__ == '__main__':
    main()
