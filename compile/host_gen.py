"""Greedy multi-token driver for a compiled iCraft decoder net on the Host backend.

正本 = `S:/work/WCIS/workenv/qwen3/host_gen.py` (304 lines, the copy that actually ran the E41 chain);
this file is that source plus one flag, `--padfix`. The tracked Qwen3 copy
`repos/Qwen3-0.6B-Instruct/compile/host_gen.py` (274 lines, md5 c9a86967b136cf31a86128d4541dd5b9) predates
the dead-input handling and cannot run a padded graph, so it is not the base for this line.

Runs one icraft-run forward per token and chains the present KV cache back into the next step.
Chaining is not a byte copy: measured on a live step, the SFB dump of a present KV tensor is
2097152 B for a @fp(16) [1, 8, 512, 128] input, i.e. icraft widens every dumped value to fp32,
so each dump is read back with the dtype implied by its size and stored again as the fp16 the
feed declares. The widening itself is lossless (all dumped values land on the fp16 grid), which
is why the round trip reproduces the SFT dump to 4.96e-05 and gives the same logits argmax.

Feed contract (read off the adapted json, and confirmed by the SFT dumps of a live step):
  v_id 0  input_ids        fp32 [1, 1]   the token of this step
  v_id 1  token_dynamic_fm fp32 [1, 3]   [new token count, slots already filled, 0]
        element 0 is sliced out (begin/end [0,0,0]-[1,1,1]) and cast into RopeDynamic;
        RopeDynamic and every SliceDynamic / CausalMask receive the whole [1, 3] tensor.
        With [1, 1, 0] the step wrote KV at slot 1, used RoPE position 1
        (dumped cos started at 0.540527 = cos(1 * 1e-6**0) for head_dim 128), and CausalMask
        opened 2 slots. So element 1 is both the write slot and the absolute position, and
        element 2 is read by nothing (feeding [1,1,0] and [1,1,1] give identical logits).
        element 1 = 0 cannot be run: the dynamic slicing blows up to a 512x larger tensor and
        op_id 4525 (Reshape) fails. The vendored export is therefore locked to the vendor's
        convention -- first token at element 1 = 1, slot 0 kept as an all-zero pad. --pos-offset 1.
  v_id 2.. past_key_values fp16 [1, 8, 512, 128], zero-filled for the positions not yet used

--padfix repairs the one defect that contract leaves behind. Because the graph never writes slot 0,
position 0 is computed as a softmax over {all-zero pad, self} instead of the pure self-attention a
prefill computes, and every later step then attends to that poisoned entry (measured on E41: step-0
logits cos -0.253818 against HF). The repair is a driver convention, not a recompile: feed token 0 a
second time and copy slot 1 into slot 0 in every cache tensor on the way in and afterwards, so the
repeat's window is {self, self} -- two identical keys and values, whose softmax is exactly
self-attention. The residue is that token 0 then carries double weight for the rest of the walk.
Step mapping: step 0 = poisoned first pass, step 1 = re-feed of token 0 (its argmax is the logits for
position 0), step k>=2 feeds prompt[k-1] with element 1 = k, so the walk occupies len(prompt)+1 slots.

Usage:
  python host_gen.py --json <adapted.json> --raw <adapted.raw> --work <dir> \
      --prompt "1427" --max-new 8
  python host_gen.py ... --padfix           # slot0=slot1 double first step (E41: gen_e41_padfix)
  python host_gen.py ... --pos-offset 1     # element 1 = filled slots + 1, what the vendor sends
  python host_gen.py ... --dry-run          # only print the feed plan, run nothing

Paths written into inputs.txt are absolute, so the launch directory does not matter; the process
cwd is set to the parent of --work because the proven single-step invocation ran from there.

Verified against a live run: this icraft-run build accepts --dump_ftmp (exactly the requested
v_ids are dumped), and SFB dumps are fp32 regardless of the declared element type -- hence
read_dump() below, which infers the dtype from file size / json element count.

Outputs under <work>: step<N>/feed/in_v<id>.ftmp, step<N>/inputs.txt,
step<N>/run/<net>/ftmpSFB/<v_id>.ftmp, step<N>/run.{out,err}.log,
tokens.txt (one id per line, prompt first then generated), steps.json (per-step argmax + timing).

Prompt phase: the --prompt tokens are fed one per step exactly like the vendor driver does, and the
argmax of every prompt step except the last is discarded -- only the last prompt token's argmax
becomes the first generated token. Appending during the prompt phase scrambles the input stream
(10-12: that bug, not the zero pad slot, is what made the 43-step chain print garbage).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DT = {'@fp(32)': np.dtype('<f4'), '@fp(16)': np.dtype('<f2'), '@fp(64)': np.dtype('<f8'),
      '@si(32)': np.dtype('<i4'), '@si(64)': np.dtype('<i8')}
EXE = 'C:/Icraft/CLI v3.39.0/bin/icraft-run.exe'
DEFAULT_STOP = '151645,151643'


def load_net(jsonpath):
    d = json.load(open(jsonpath, encoding='utf-8'))
    ops = d['ops']
    producer = {}
    for op in ops:
        for o in op.get('outputs') or []:
            producer[o['v_id']] = op

    ins = sorted(ops[0]['outputs'], key=lambda o: o['v_id'])
    assert ops[0]['_type_key'].endswith('Input')
    outs = ops[-1]['inputs']
    assert ops[-1]['_type_key'].endswith('Output')

    def nbytes(spec):
        return DT[spec['dtype']['element_dtype']].itemsize * math.prod(spec['dtype']['shape'])

    consumed = {i['v_id'] for op in ops[1:] for i in (op.get('inputs') or [])}
    ctrl = [o for o in ins if o['v_id'] in (0, 1)]
    # a padded graph declares cache inputs that nothing reads; they still have to be fed, because the
    # runtime checks network_inputs_size == tensor_inputs_size before it dispatches a single op
    dead = [o for o in ins if o['v_id'] > 1 and o['v_id'] not in consumed]
    past = [o for o in ins if o['v_id'] > 1 and o['v_id'] in consumed]

    # present output -> the past input it was built from, by walking back to its SliceDynamic
    pair = {}
    for o in outs:
        if o['v_id'] in (0, 1):
            continue
        cur, depth = o['v_id'], 0
        while depth < 30:
            op = producer.get(cur)
            if op is None:
                break
            tk = op['_type_key']
            if tk.endswith('SliceDynamic'):
                pair[o['v_id']] = op['inputs'][-1]['v_id']
                break
            if not op.get('inputs'):
                break
            cur = op['inputs'][0]['v_id']
            depth += 1

    if len(pair) != len(past):
        raise SystemExit('KV pairing incomplete: %d outputs matched, %d past inputs' % (len(pair), len(past)))
    if sorted(pair.values()) != sorted(o['v_id'] for o in past):
        raise SystemExit('KV pairing is not a bijection over the past inputs')

    logits = [o for o in outs if o['v_id'] not in pair]
    if len(logits) != 1:
        raise SystemExit('expected exactly one non-KV output, got %s' % [o['v_id'] for o in logits])
    return {'name': d['name'], 'ctrl': ctrl, 'past': past, 'dead': dead, 'pair': pair,
            'nbytes': {o['v_id']: nbytes(o) for o in ins + outs},
            'dt': {o['v_id']: DT[o['dtype']['element_dtype']] for o in ins + outs},
            'logits': logits[0], 'shape': logits[0]['dtype']['shape']}


def write_ctrl(feed_dir, net, token, dyn):
    paths = []
    for spec in net['ctrl']:
        arr = np.zeros(net['nbytes'][spec['v_id']] // np.dtype(net['dt'][spec['v_id']]).itemsize,
                       dtype=net['dt'][spec['v_id']])
        if spec['v_id'] == 0:
            arr[0] = token
        else:
            arr[:len(dyn)] = np.asarray(dyn, dtype=net['dt'][spec['v_id']])
        p = os.path.join(feed_dir, 'in_v%d.ftmp' % spec['v_id'])
        arr.tofile(p)
        paths.append(p)
    return paths


def seed_past(feed_dir, net, seed_dir=None):
    paths = []
    for spec in net['past']:
        p = os.path.join(feed_dir, 'in_v%d.ftmp' % spec['v_id'])
        src = os.path.join(seed_dir, 'in_v%d.ftmp' % spec['v_id']) if seed_dir else None
        if src:
            if not os.path.exists(src):
                raise SystemExit('--seed-dir given but %s is missing' % src)
            if os.path.getsize(src) != net['nbytes'][spec['v_id']]:
                raise SystemExit('%s is %d bytes, feed wants %d'
                                 % (src, os.path.getsize(src), net['nbytes'][spec['v_id']]))
            arr = np.fromfile(src, dtype=net['dt'][spec['v_id']])
        else:
            arr = np.zeros(net['nbytes'][spec['v_id']] // np.dtype(net['dt'][spec['v_id']]).itemsize,
                           dtype=net['dt'][spec['v_id']])
        arr.tofile(p)
        paths.append(p)
    return paths


def seed_dead(feed_dir, net):
    """Zero-fill the declared-but-unread cache inputs; they sit above every past v_id, so they append last."""
    paths = []
    for spec in net['dead']:
        p = os.path.join(feed_dir, 'in_v%d.ftmp' % spec['v_id'])
        np.zeros(net['nbytes'][spec['v_id']] // np.dtype(net['dt'][spec['v_id']]).itemsize,
                 dtype=net['dt'][spec['v_id']]).tofile(p)
        paths.append(p)
    return paths


def read_dump(path, shape):
    """Read an SFB dump. icraft widens dumped values to fp32, so the dtype comes from the size."""
    numel = math.prod(int(s) for s in shape)
    size = os.path.getsize(path)
    if size % numel:
        raise SystemExit('%s is %d bytes, not a multiple of %d elements' % (path, size, numel))
    dt = {2: '<f2', 4: '<f4', 8: '<f8'}.get(size // numel)
    if dt is None:
        raise SystemExit('%s: %d bytes for %d elements, no fp dtype matches' % (path, size, numel))
    return np.fromfile(path, dtype=dt)


def chain_past(feed_dir, net, prev_dump_dir, padfix=False):
    paths = []
    for spec in net['past']:
        src = None
        for out_v, past_v in net['pair'].items():
            if past_v == spec['v_id']:
                src = os.path.join(prev_dump_dir, '%d.ftmp' % out_v)
        arr = read_dump(src, spec['dtype']['shape'])
        want = DT[spec['dtype']['element_dtype']]
        if arr.dtype != want:
            if not np.all(np.isfinite(arr.astype(want))):
                raise SystemExit('%s has values outside the feed dtype range' % src)
            arr = arr.astype(want)
        if padfix:
            arr = arr.reshape(spec['dtype']['shape'])
            arr[:, :, 0, :] = arr[:, :, 1, :]
        p = os.path.join(feed_dir, 'in_v%d.ftmp' % spec['v_id'])
        arr.tofile(p)
        paths.append(p)
    return paths


def run_step(args, net, inputs_txt, step_dir):
    extra = [int(v) for v in args.extra_dump.split(',') if v.strip()]
    dump_ids = ','.join(str(v) for v in [net['logits']['v_id']] + sorted(net['pair']) + extra)
    log_dir = os.path.join(step_dir, 'run')
    os.makedirs(log_dir, exist_ok=True)
    cmd = [args.exe,
           '--json', args.json, '--raw', args.raw,
           '--input', inputs_txt,
           '--backends', 'Host',
           '--dump_output', '--dump_format', args.dump_format.upper(), '--dump_ftmp', dump_ids,
           '--log_path', log_dir, '--log_time']
    t0 = time.time()
    with open(os.path.join(step_dir, 'run.out.log'), 'wb') as fo, \
            open(os.path.join(step_dir, 'run.err.log'), 'wb') as fe:
        rc = subprocess.call(cmd, stdout=fo, stderr=fe,
                             cwd=os.path.dirname(os.path.abspath(args.work)))
    dt = time.time() - t0
    dump_dir = os.path.join(log_dir, net['name'], 'ftmp' + args.dump_format.upper())
    if rc != 0:
        sys.stderr.write(open(os.path.join(step_dir, 'run.err.log'), encoding='utf-8',
                              errors='replace').read()[-2000:])
        raise SystemExit('icraft-run rc=%d at step %s' % (rc, step_dir))
    return dt, dump_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--prompt', required=True, help='comma separated token ids of the prompt')
    ap.add_argument('--max-new', type=int, default=8)
    ap.add_argument('--pos-offset', type=int, choices=[0, 1], default=1,
                    help='element 1 = filled slots + offset; 0 makes the graph fail at op_id 4525')
    ap.add_argument('--tdf', default='', help='force the whole token_dynamic_fm triple, e.g. "1,1,1", '
                                              'ignoring --pos-offset; for probing the graph only')
    ap.add_argument('--dump-format', default='SFB', help='SFB for chaining, SFT when you want to read '
                                                         'the dumps as text in a probe')
    ap.add_argument('--stop', default=DEFAULT_STOP)
    ap.add_argument('--seed-dir', default='', help='directory of in_v<id>.ftmp cache feeds used for '
                                                   'step 0 instead of zeros; later steps still chain')
    ap.add_argument('--past-base', type=int, default=0, help='slots already occupied before step 0')
    ap.add_argument('--extra-dump', default='', help='comma separated extra v_ids to dump, e.g. "180,37923"')
    ap.add_argument('--padfix', action='store_true',
                    help='re-feed token 0 with slot 0 = slot 1 so position 0 gets pure self-attention')
    ap.add_argument('--exe', default=EXE)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    net = load_net(args.json)
    prompt = [int(t) for t in args.prompt.split(',') if t.strip()]
    stops = {int(t) for t in args.stop.split(',') if t.strip()}
    walk = ([prompt[0]] + prompt) if args.padfix else list(prompt)
    seq = list(prompt)
    feed = list(walk)
    os.makedirs(args.work, exist_ok=True)
    print('net', net['name'], 'ctrl inputs', [o['v_id'] for o in net['ctrl']],
          'past', len(net['past']), 'logits v_id', net['logits']['v_id'], net['logits']['dtype']['shape'])
    print('prompt', len(prompt), 'tokens', prompt)
    if args.padfix:
        print('padfix walk', len(walk), 'steps (token', prompt[0], 'repeated at step 1)')
    if args.dry_run:
        print('dump ids', [net['logits']['v_id']] + sorted(net['pair']))
        print('pair sample', list(net['pair'].items())[:4])
        return

    record = []
    for i in range(len(walk) + args.max_new):
        step_dir = os.path.join(args.work, 'step%d' % i)
        feed_dir = os.path.join(step_dir, 'feed')
        os.makedirs(feed_dir, exist_ok=True)
        token = feed[i]
        past_len = args.past_base + i + args.pos_offset - (1 if (args.padfix and i >= 1) else 0)
        if past_len >= 512:
            print('cache full at step %d' % i)
            break
        dyn = ([float(x) for x in args.tdf.split(',')] if args.tdf
               else [1.0, float(past_len), 0.0])
        paths = write_ctrl(feed_dir, net, token, dyn)
        prev = os.path.join(args.work, 'step%d' % (i - 1), 'run', net['name'],
                            'ftmp' + args.dump_format.upper())
        paths += (chain_past(feed_dir, net, prev, args.padfix) if i
                  else seed_past(feed_dir, net, args.seed_dir or None))
        paths += seed_dead(feed_dir, net)
        inputs_txt = os.path.join(step_dir, 'inputs.txt')
        open(inputs_txt, 'w', encoding='utf-8', newline='\n').write(
            '\n'.join(os.path.abspath(p) for p in paths) + '\n')

        dt, dump_dir = run_step(args, net, inputs_txt, step_dir)
        lg_path = os.path.join(dump_dir, '%d.ftmp' % net['logits']['v_id'])
        lg = read_dump(lg_path, net['logits']['dtype']['shape'])
        nxt = int(np.argmax(lg))
        np.save(os.path.join(step_dir, 'logits.npy'), lg.astype(np.float32))
        if i >= len(walk) - 1:
            seq.append(nxt)
            feed.append(nxt)
        record.append({'step': i, 'fed': token, 'past_len': past_len, 'argmax': nxt,
                       'logits_min': float(lg.min()), 'logits_max': float(lg.max()),
                       'seconds': round(dt, 1)})
        print('step %3d fed %-7d past %-4d -> %-7d  (%.1fs, logits %.3f..%.3f)'
              % (i, token, past_len, nxt, dt, lg.min(), lg.max()))
        sys.stdout.flush()
        json.dump(record, open(os.path.join(args.work, 'steps.json'), 'w', encoding='utf-8'), indent=1)
        open(os.path.join(args.work, 'tokens.txt'), 'w', encoding='utf-8', newline='\n').write(
            '\n'.join(str(t) for t in seq) + '\n')
        if i >= len(walk) - 1 and nxt in stops:
            print('stop token %d at step %d' % (nxt, i))
            break

    print('sequence', seq)


if __name__ == '__main__':
    main()
