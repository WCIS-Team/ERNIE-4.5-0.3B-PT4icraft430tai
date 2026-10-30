"""Delete the attention from ERNIE's real graph to see whether anything else parses.

Where this comes from. Every node-local variable has been eliminated on the tiled lineage -- head
geometry copied to Qwen3's exact 16Q/8KV/head_dim 128, rope cache rows 40960, rotary_interleaved 0,
do_rotary 0, the Split sizes operand as a Constant node, 54 identity Reshapes on the GQA slots, and
finally Q/K/V handed in as fresh graph inputs (e_v10) -- and the parse still stops on the first
GroupQueryAttention: the log's last lowered op is op_id 8 (k_head_concat) and the next line is the
access-violation/bad-allocation crash. A standalone single-GQA harness was built to isolate the node
(mk_minimal_gqa.py) but all three of its variants, including the one carrying Qwen3's own attributes,
crash identically, so the harness is missing something the parser needs and proves nothing.

The remaining question is directional and cheap to answer: is the GQA node the only thing icraft
cannot lower, or is ERNIE fatal somewhere else too and the GQA just happens to be reached first?
This script answers it by removing the attention entirely. Each layer's o_proj MatMul is rewired to
consume the qkv Split's Q output directly -- widths already agree, both are 2048 wide, so the
matmul stays dimensionally honest -- and the 18 GroupQueryAttention nodes plus the 36 present.*
graph outputs go with them.

Reading the result. If this graph parses, the rest of ERNIE is fine and the only blocker is this one
node's lowering; that justifies hand-decomposing GQA into the primitive sequence Qwen3 lowers to
(Expand/Cast/RopeDynamic/Reshape/Transpose/Slice/Neg/Concat/Multiply/Add/Matmul/Softmax plus the
cache Concat), because every one of those is already proven in this pipeline. If it still crashes,
the blocker is elsewhere in the graph and decomposing the attention would not help -- the whole
model needs a different route.

The KV-cache inputs (past_key_values.*) stay declared as graph inputs so the parser input contract
that was already verified identical to Qwen3's is not also being changed; nothing downstream consumes
them, which is fine for a parse-only probe.
"""
import argparse
import os
import sys

import onnx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='e_v14')
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, 'fmodel', args.src)
    dst_dir = os.path.join(ROOT, 'fmodel', args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    m = onnx.load(os.path.join(src_dir, 'model.onnx'), load_external_data=False)
    g = m.graph
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n

    gqa = [n for n in g.node if n.op_type == 'GroupQueryAttention']
    if not gqa:
        raise SystemExit('no GroupQueryAttention in %s' % args.src)

    rewired = 0
    for node in gqa:
        q = node.input[0]
        out = node.output[0]
        consumers = [n for n in g.node if out in n.input]
        if len(consumers) != 1:
            raise SystemExit('%s has %d consumers, expected 1' % (node.name, len(consumers)))
        c = consumers[0]
        c.input[list(c.input).index(out)] = q
        rewired += 1
    for node in gqa:
        g.node.remove(node)
    print('GQA nodes removed: %d; consumers rewired to Q: %d' % (len(gqa), rewired))

    dropped_outputs = 0
    for v in list(g.output):
        if v.name.startswith('present.'):
            g.output.remove(v)
            dropped_outputs += 1
    print('present.* graph outputs dropped: %d; remaining outputs: %s'
          % (dropped_outputs, [v.name for v in g.output]))

    # nothing may reference a removed tensor any more
    live = {v.name for v in list(g.input) + list(g.output) + list(g.value_info)}
    inits = {i.name for i in g.initializer}
    produced = set()
    for n in g.node:
        produced.update(o for o in n.output if o)
    dead = live | produced | inits
    bad = [(n.name, i) for n in g.node for i in n.input
           if i and i not in dead]
    if bad:
        raise SystemExit('dangling operands after removal: %s' % bad[:6])
    # value_info describing removed producers would be stale; keep only live ones
    for v in list(g.value_info):
        if v.name.startswith('present.'):
            g.value_info.remove(v)

    onnx.save(m, os.path.join(dst_dir, 'model.onnx'))
    link = os.path.join(dst_dir, 'model.onnx.data')
    if not os.path.exists(link):
        os.link(os.path.join(src_dir, 'model.onnx.data'), link)
        print('hardlinked data')
    for extra in ('genai_config.json', 'config.json'):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst_dir, extra), 'w', encoding='utf-8',
                 newline='\n').write(open(s, encoding='utf-8').read())

    chk = onnx.load(os.path.join(dst_dir, 'model.onnx'), load_external_data=False)
    print('nodes %d  GQA %d  op_types: %s'
          % (len(chk.graph.node),
             sum(1 for n in chk.graph.node if n.op_type == 'GroupQueryAttention'),
             sorted({n.op_type for n in chk.graph.node})))
    print('DONE_DROPATTN')


if __name__ == '__main__':
    main()
