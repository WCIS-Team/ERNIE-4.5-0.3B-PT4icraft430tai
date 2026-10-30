"""Cut one GroupQueryAttention loose from its producers and see if icraft still lowers it.

State of play: with the qkv unpacked (split_qkv) the parser gets to op_id 8 and then dies entering
the first GroupQueryAttention -- access violation surfaced as "bad allocation", before it prints a
single line for that node. Nothing about the node itself explains it any more, because e_v8's layer-0
GQA now matches q3i_v1's (which compiles) attribute for attribute and shape for shape:

    num_heads 16 / kv_num_heads 8 / head_dim 128 / Q 2048 wide / K,V 1024 wide / scale 0.0883883
    / softcap 0 / local_window_size -1 / 12 input slots with 3 empty / past [b,8,past,128]

and these were each already eliminated on the real graph: rope cache rows 131072 -> 40960,
rotary_interleaved 1 -> 0, do_rotary 1 -> 0, sizes operand initializer -> Constant node,
toml max_prompt/max_tokens 32/512 -> 8/64.

That leaves one axis untouched: WHO feeds the node. q3i_v1 hands Q and K in from Reshapes (the
per-head q_norm/k_norm chains) and V from a Split output_2; ours hands Q from a Split output_0 and
K,V from Concats. Replace those three producers by plain graph inputs and the GQA is lowered with no
provenance to inspect at all.

Reading the result:
  * op_id count climbs past 9 and the crash moves to a later GQA  -> the producer chain is the cause;
    patch layer 0 only, so layer 1 (still fed by Split/Concat) is the next thing the parser meets.
  * still dies at op_id 9                                        -> the node's own context is the cause
    (cache contract, mask slots, or graph-level metadata), and the next probe is the same surgery on
    q3i_v1 as a control, to see whether decoupling is itself fatal to icraft.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile', 'fmodel')
SLOT_NAME = {0: 'q', 1: 'k', 2: 'v'}


def dims_of(g, name):
    for coll in (g.value_info, g.input, g.output):
        for v in coll:
            if v.name == name:
                return ([d.dim_value if d.HasField('dim_value') else d.dim_param
                         for d in v.type.tensor_type.shape.dim],
                        v.type.tensor_type.elem_type)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v8')
    ap.add_argument('--dst', required=True)
    ap.add_argument('--layer', type=int, default=0,
                    help='which attention layer to decouple (-1 = every layer)')
    ap.add_argument('--slots', default='0,1,2')
    ap.add_argument('--graph', default=None,
                    help='absolute model.onnx of a graph outside the ernie fmodel root')
    args = ap.parse_args()

    if args.graph:
        src_model = args.graph
        src_dir = os.path.dirname(src_model)
    else:
        src_dir = os.path.join(ROOT, args.src)
        src_model = os.path.join(src_dir, 'model.onnx')
    dst_dir = os.path.join(ROOT, args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(src_model, load_external_data=False)
    g = m.graph
    slots = [int(s) for s in args.slots.split(',')]
    gqas = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    picked = [n for i, n in enumerate(gqas)
              if args.layer < 0 or i == args.layer]
    print('GQA nodes %d, decoupling %d, slots %s' % (len(gqas), len(picked), slots))

    for n in picked:
        tag = 'L%d' % gqas.index(n) if args.layer < 0 else ''
        for s in slots:
            old = n.input[s]
            dims, dt = dims_of(g, old)
            if dims is None:
                raise SystemExit('no shape known for %s (slot %d) -- cannot decouple' % (old, s))
            name = 'dbg_%s%s_in%d' % (SLOT_NAME[s], tag, s)
            if any(v.name == name for v in g.input):
                raise SystemExit('input %s already exists' % name)
            g.input.append(onnx.helper.make_tensor_value_info(name, dt, dims))
            n.input[s] = name
            print('  slot %d: %s %s -> graph input %s' % (s, old, dims, name))

    out_path = os.path.join(dst_dir, 'model.onnx')
    onnx.save(m, out_path)
    link = os.path.join(dst_dir, 'model.onnx.data')
    if not os.path.exists(link):
        os.link(os.path.join(src_dir, 'model.onnx.data'), link)
        print('hardlinked data')
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst_dir, extra), 'w', encoding='utf-8', newline='\n').write(
                open(s, encoding='utf-8').read())

    chk = onnx.load(out_path, load_external_data=False)
    cg = chk.graph
    first = [n for n in cg.node if n.op_type == 'GroupQueryAttention'][0]
    print('first GQA inputs now: %s' % [i for i in first.input[:3]])
    print('graph inputs: %d' % len(cg.input))
    print('onnx', os.path.getsize(out_path), 'bytes')
    print('DONE_DECOUPLE')


if __name__ == '__main__':
    main()
