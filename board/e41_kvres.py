#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV-resident experiment: can the 36 KV tensors stay on plddr across steps?

E41KV=off|inplace|copy|bindonly
               off      control: host KV, two-arg forward into our own chunks
               inplace  36 chunks; the SAME chunk is the past input and the bound
                        present output -- zero copies, but present/past alias.
               copy     72 chunks: past input and bound present output are DIFFERENT
                        buffers, and each step does a device->device copyFrom of
                        36 x 1 MiB instead of the host round-trip. No aliasing.
               bindonly past KV still come from host; only the present outputs are
                        bound to plddr. Isolates "cost of binding" from
                        "cost of reading the history off PL DDR".
               Those v_ids are then fed/returned as Tensors wrapping the chunk, so
               the host never sees the KV.

Built-in discriminator: with E41KV=inplace/copy the host numpy KV arrays are never
written and never read. If the runtime still uploaded them, the chunks would be
re-zeroed every step and the chain would collapse -- so "chain == control" proves
the upload is gone.

E41PAD=dev|off|host   slot0=slot1 fix; dev = chunk.copyFrom on device, host = read back
E41DEAD=host|chunk    the 4 unconsumed KV declarations
E41RESET=1|0          dev.reset(1) per step (parity knob with the control path)
E41ALIGN=nnn          malloc alignment for every buffer we own (hardop needs phy %4096==0)
"""
import datetime as DT
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.append("/root/qwen2/modelzoo_utils")
import icraft                      # noqa: E402
from icraft import xir, xrt        # noqa: E402
from icraft import zg330backend    # noqa: E402
from pyrtutils.icraft_utils import openDevice, initSession   # noqa: E402

Tensor = xrt.Tensor
JSON_ = os.environ.get("E41_JSON", "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.json")
RAW_ = os.environ.get("E41_RAW", "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.raw")
CFG = os.environ.get("E41P_CFG", "kvres")
OCM_OPT = int(os.environ.get("E41P_OCM", "1"))
MERGE = os.environ.get("E41P_MERGE", "on")
ETM = os.environ.get("E41P_ETM", "on")
STEPS = int(os.environ.get("E41P_STEPS", "12"))
KV = os.environ.get("E41KV", "inplace")
PAD = os.environ.get("E41PAD", "dev")
DEAD = os.environ.get("E41DEAD", "host")
RESET = os.environ.get("E41RESET", "1") == "1"
PREFIX = [int(x) for x in os.environ.get("E41P_PREFIX", "").replace(" ", "").split(",") if x]
OUT = os.environ.get("E41P_OUT", "/mnt/data/ernie/kv/%s.txt" % CFG)
OUT_JSON = os.environ.get("E41P_OUTJSON", "")
CMA_FLOOR = int(os.environ.get("E41_CMA_FLOOR_KB", "4000"))
# hardop.cpp:107 aborts the two-arg forward() on any output whose phy_addr is not
# a multiple of 4096, and Tensor.mallocOn(region) has no alignment knob at all, so
# every buffer here comes from reg.malloc(size, auto_free=False, alignment=ALIGN).
ALIGN = int(os.environ.get("E41ALIGN", "4096"))
# defaultMemRegion is plddr (PL-side DDR); the runtime's own INPUT/OUTPUT segments
# are not there, and both forward() and a plain read() are several times slower on
# our chunks -- so let the region be chosen, to see which memory the gate is in.
REG = os.environ.get("E41REG", "default")
# E41PROF=1 turns on Session.enableTimeProfile(True) BEFORE apply() and dumps
# timeProfileResults() twice (after 2 steps / at the end).  Composition only: the
# profiler itself moves time (ledger 28.4: forward 393 -> 434 ms), so these numbers
# answer "which entry is the 58.8 ms memcpy", not "how fast is this gear".
PROF = os.environ.get("E41PROF", "0") == "1"
# E41P_LOGDIR=<dir> additionally saves the full float32 logits of every step as
# <dir>/<CFG>_step<past_len>.npy.  Used to compare two 件 on the same fed tokens:
# argmax alone cannot tell "same numbers, borderline flip" from "different numbers".
LOGDIR = os.environ.get("E41P_LOGDIR", "")

NH, NS, HD = 8, 512, 128           # [1,8,512,128] fp16 -- defaults, rebound from the graph
SLOT_B = HD * 2                    # bytes per (head, slot) = head_dim * itemsize
HEAD_STRIDE_B = NS * SLOT_B        # 131072 B per head
KV_B = NH * HEAD_STRIDE_B          # 1048576 B per tensor

_LOG = open(OUT, "w", encoding="utf-8")


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    _LOG.write(line + "\n")
    _LOG.flush()


def use_graph_geom(kv_shape, itemsize):
    """Rebind the KV geometry to what the compiled network actually declares.

    NS comes from the toml's max_tokens_length at compile time, so a shorter-cache
    件 has a different NS and every buffer this file sizes must follow the graph.
    On E41 the shape equals the constants above, so it is a no-op there.
    """
    global NH, NS, HD, SLOT_B, HEAD_STRIDE_B, KV_B
    NH, NS, HD = int(kv_shape[1]), int(kv_shape[2]), int(kv_shape[3])
    SLOT_B = HD * itemsize
    HEAD_STRIDE_B = NS * SLOT_B
    KV_B = NH * HEAD_STRIDE_B
    say("GEOM kv_shape=%r itemsize=%d -> NH=%d NS=%d HD=%d SLOT_B=%d KV_B=%d" % (
        kv_shape, itemsize, NH, NS, HD, SLOT_B, KV_B))


def meminfo():
    want = ("MemAvailable", "CmaFree")
    out = {}
    for ln in open("/proc/meminfo"):
        k = ln.split(":")[0]
        if k in want:
            out[k] = ln.split()[1]
    return " ".join("%s=%s kB" % (k, out.get(k, "?")) for k in want)


def cma_free():
    for ln in open("/proc/meminfo"):
        if ln.startswith("CmaFree"):
            return int(ln.split()[1])
    return -1


def read_stat(keys, path="/proc/self/status"):
    out = {}
    for ln in open(path):
        k = ln.split(":")[0]
        if k in keys:
            out[k] = ln.split()[1] + ln.split()[2]
    return " ".join("%s=%s" % (w, out.get(w, "?")) for w in keys)


def load_graph_map(jsonpath):
    """present output index -> past input v_id, plus the unconsumed input v_ids."""
    d = json.load(open(jsonpath, encoding="utf-8"))
    ops = d["ops"]
    producer = {}
    for op in ops:
        for o in op.get("outputs") or []:
            producer[o["v_id"]] = op
    ins = sorted(ops[0]["outputs"], key=lambda o: o["v_id"])
    outs = ops[-1]["inputs"]
    consumed = set()
    for op in ops[1:]:
        for i in (op.get("inputs") or []):
            consumed.add(i["v_id"])
    dead = [o["v_id"] for o in ins if o["v_id"] > 1 and o["v_id"] not in consumed]
    pair = {}
    for idx, o in enumerate(outs):
        if o["v_id"] in (0, 1):
            continue
        cur, depth = o["v_id"], 0
        while depth < 30:
            op = producer.get(cur)
            if op is None:
                break
            if op["_type_key"].endswith("SliceDynamic"):
                pair[idx] = op["inputs"][-1]["v_id"]
                break
            if not op.get("inputs"):
                break
            cur = op["inputs"][0]["v_id"]
            depth += 1
    return {"ins": [o["v_id"] for o in ins], "pair": pair, "dead": dead,
            "out_vids": [o["v_id"] for o in outs]}


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def summarise(name, vals):
    v = [x * 1000.0 for x in vals]
    say("SEG %-16s n=%d min=%.2f med=%.2f p95=%.2f max=%.2f ms" % (
        name, len(v), min(v), pct(v, 0.5), pct(v, 0.95), max(v)))


def spec_bytes(s):
    t = s.tensorType()
    n = 1
    for d in list(t.shape):
        n *= int(d)
    return n * (4 if "fp(32)" in str(t.element_dtype) else 2)


def num(v):
    try:
        return int(v)
    except Exception:
        return None


def dump_seg(zg, attr, label):
    try:
        segmap = getattr(zg, attr)
    except Exception as e:
        say("SEGMAP %s %s -> %r" % (label, attr, e))
        return
    tot = 0
    parts = []
    for st, seg in segmap.items():
        b = num(seg.byte_size) if hasattr(seg, "byte_size") else None
        if b is None:
            parts.append("%s=?" % st)
            continue
        tot += b
        parts.append("%s=%d" % (st, b))
    say("SEGMAP %s %s TOTAL=%d B (%.1f MiB)" % (label, " ".join(parts), tot, tot / 1048576.0))


class Watchdog(threading.Thread):
    """Abort THIS process only if CMA drops under the floor; never touches any
    other process on the board."""

    def __init__(self, floor_kb):
        threading.Thread.__init__(self)
        self.daemon = True
        self.floor = floor_kb
        self.low = None
        self.stop = False

    def run(self):
        while not self.stop:
            f = cma_free()
            if self.low is None or f < self.low:
                self.low = f
            if 0 <= f < self.floor:
                say("WATCHDOG CmaFree=%d kB < floor %d -- aborting self" % (f, self.floor))
                _LOG.flush()
                os._exit(9)
            time.sleep(1.0)


class E41(object):
    def __init__(self):
        self.kv_on = (KV != "off")
        # bindonly answers "which side costs the extra ~110 ms of forward": outputs
        # are bound to plddr chunks, but the past KV are still uploaded from host.
        self.feed_dev = (KV in ("inplace", "copy"))
        t0 = time.time()
        self.net = xir.Network.CreateFromJsonFile(JSON_)
        self.net.lazyLoadParamsFromFile(RAW_)
        say("json+raw %.1f s" % (time.time() - t0))
        self.gmap = load_graph_map(JSON_)
        self.pair = self.gmap["pair"]
        self.out_vids = self.gmap["out_vids"]
        say("pair=%d dead=%s out_vids=%d" % (len(self.pair), self.gmap["dead"],
                                             len(self.out_vids)))
        if len(self.pair) != 36:
            raise SystemExit("KV pairing incomplete")
        self.specs = list(self.net.ops[0].outputs)
        self.ospecs = list(self.net.ops[-1].inputs)
        self.vid_of_spec = [s.v_id for s in self.specs]
        self.host_feed = []
        for s in self.specs:
            t = s.tensorType()
            dt = np.float32 if "fp(32)" in str(t.element_dtype) else np.float16
            self.host_feed.append(np.zeros(list(t.shape), dtype=dt))
        kv_shape = list(self.specs[2].tensorType().shape)
        if len(kv_shape) != 4 or int(kv_shape[0]) != 1:
            raise SystemExit("unexpected KV input shape %r on v2" % (kv_shape,))
        use_graph_geom(kv_shape, self.host_feed[2].itemsize)

        t_mid = time.time()
        self.dev = openDevice("zg330", "192.168.125.143", True, False)
        self.sess = initSession("zg330", self.net, self.dev, OCM_OPT,
                                mmuMode=True, open_speedmode=(MERGE == "on"),
                                open_compressFtmp=(ETM == "on"))
        self.zg = zg330backend.ZG330Backend(self.sess.backends[0])
        say("mid %.1f s (graph map+openDevice+initSession)" % (time.time() - t_mid))
        dump_seg(self.zg, "logic_segment_map", "logic(before connect)")

        self.reg = (self.dev.defaultMemRegion() if REG == "default"
                    else self.dev.getMemRegion(REG))
        say("region=%s default same_as plddr=%r udma=%r ocm=%r" % (
            REG,
            self.reg.same_as(self.dev.getMemRegion("plddr")),
            self.reg.same_as(self.dev.getMemRegion("udma")),
            self.reg.same_as(self.dev.getMemRegion("ocm"))))
        self.oidx_sorted = sorted(self.pair)
        self.chunks = []
        self.out_chunks = []
        self.copy_pairs = []
        self.pad_chunks = []
        self.chunk_of_in_vid = {}
        self.chunk_of_oidx = {}
        self.dead_chunks = {}
        if self.kv_on:
            self._connect()
        if PROF:
            try:
                self.sess.enableTimeProfile(True)
                say("enableTimeProfile(True) ok time_profile=%r" % (
                    getattr(self.sess, "time_profile", "?"),))
            except Exception as e:
                say("enableTimeProfile FAILED %r" % (e,))
        t0 = time.time()
        try:
            self.sess.apply()
        except Exception as e:
            say("APPLY FAILED after %.1f s: %r" % (time.time() - t0, e))
            raise
        say("apply %.1f s" % (time.time() - t0))
        say("CAP ocmopt=%r is_mergeHardop=%r is_applied=%r | %s" % (
            getattr(self.zg, "ocmopt", "?"), getattr(self.zg, "is_mergeHardop", "?"),
            getattr(self.zg, "is_applied", "?"), meminfo()))
        dump_seg(self.zg, "phy_segment_map", "phy(after apply)")
        self.outs, self.n_chunk_out = self.build_outs()
        say("OUT tensors built n=%d (%d on bound KV chunks, %d own chunks align=%d)" % (
            len(self.outs), self.n_chunk_out, len(self.outs) - self.n_chunk_out, ALIGN))
        self.align_report()
        if self.kv_on:
            vi = self.zg.value_info
            say("value_info len=%d; bindable? token=%r logits=%r present0=%r past_kv2=%r" % (
                len(vi), 1 in vi, self.out_vids[0] in vi,
                self.out_vids[self.oidx_sorted[0]] in vi, 2 in vi))
            same = sum(1 for oidx, ck in self.chunk_of_oidx.items()
                       if self.outs[oidx].chunk().same_as(ck))
            say("OUT chunk == bound chunk for %d/%d KV tensors (in-place with the past input)" % (
                same, len(self.oidx_sorted)))

    def _mk(self, n):
        try:
            return self.reg.malloc(n, False, ALIGN)
        except Exception as e:
            say("MALLOC %d B on region=%s FAILED: %r | %s" % (n, REG, e, meminfo()))
            raise SystemExit(2)

    def _connect(self):
        """One chunk per KV tensor.

        Binding is only possible on the OUTPUT side: userConnectNetwork() looks
        the v_id up in the backend's value_info map and the 40 past-KV input
        v_ids (2..41) are not in it -- they have no data segment of their own,
        so binding them raises IndexError('_Map_base::at').  The 36 present v_ids
        and the logits v_id are.  The past inputs are instead fed as
        Tensor(dtype, chunk), i.e. already resident on the same plddr address.

        copy mode keeps two buffers per tensor: the bound present output and a
        separate past input, and the step loop copies output->input on device.
        """
        t0 = time.time()
        nb = 0
        for k, oidx in enumerate(self.oidx_sorted):
            pvid = self.pair[oidx]
            ck_out = self._mk(KV_B)
            self.zg.userConnectNetwork(ck_out, self.out_vids[oidx])
            nb += 1
            self.chunks.append(ck_out)
            self.chunk_of_oidx[oidx] = ck_out
            if KV == "copy":
                ck_in = self._mk(KV_B)
                self.chunks.append(ck_in)
                self.copy_pairs.append((ck_in, ck_out))
                self.pad_chunks.append(ck_in)
                self.chunk_of_in_vid[pvid] = ck_in
            elif KV == "inplace":
                self.pad_chunks.append(ck_out)
                self.chunk_of_in_vid[pvid] = ck_out
        if DEAD == "chunk":
            for vid in self.gmap["dead"]:
                ck = self._mk(KV_B)
                self.dead_chunks[vid] = ck
                self.chunks.append(ck)
            nb += 0  # dead inputs have no bindable segment either
        say("CONNECT %d chunks / %d output binds in %.2f s | %s" % (
            len(self.chunks), nb, time.time() - t0, meminfo()))
        zero = np.zeros(NH * NS * HD, dtype=np.float16)
        for ck in self.chunks:
            ck.write(0, zero, KV_B)
        say("PRIMED %d chunks with zeros" % len(self.chunks))

    def build_outs(self):
        """Output tensors are built once: forward() writes into them in place.

        Constraint found by running (2026-11-02): the two-arg forward() calls
        MemPtr::addr() on every output, which aborts with
        "Check failed: ptype_ != PtrType::CPTR / The MemPtr is of CPTR type!"
        for a tensor made by Tensor(buffer, dtype) -- that constructor keeps a
        raw Python pointer.  And hardop.cpp:107 then aborts on
        "output tensor physic addr: NNN is not align to 4096" for anything from
        Tensor.mallocOn(region).  So every output is a chunk we malloc() with an
        explicit alignment; nothing is left to mallocOn.
        """
        outs = []
        n_chunk = 0
        for oidx, s in enumerate(self.ospecs):
            ck = self.chunk_of_oidx.get(oidx)
            if ck is None:
                ck = self._mk(spec_bytes(s))
                self.out_chunks.append(ck)
            else:
                n_chunk += 1
            outs.append(Tensor(s.tensorType(), ck))
        return outs, n_chunk

    def align_report(self):
        """Does the alignment we asked for survive into value_info.phy_addr?"""
        vi = self.zg.value_info
        rows = []
        bad = 0
        for oidx, s in enumerate(self.ospecs):
            vid = self.out_vids[oidx]
            try:
                phy = int(vi[vid].phy_addr)
            except Exception as e:
                rows.append("%d:ERR(%r)" % (vid, e))
                bad += 1
                continue
            ck = self.outs[oidx].chunk()
            try:
                beg = int(ck.begin.addr())
            except Exception as e:
                beg = None
                rows.append("%d:phy=%d?begin_err(%r)" % (vid, phy, e))
            if phy % 4096:
                bad += 1
            rows.append("v%d phy=%d(m%d) beg=%s" % (
                vid, phy, phy % 4096, "?" if beg is None else "%d(m%d)" % (beg, beg % 4096)))
        say("ALIGN asked=%d; outputs not phy-aligned=%d/%d" % (ALIGN, bad, len(self.outs)))
        say("ALIGN sample %s" % " | ".join(rows[:3] + rows[-2:]))

    def build_ins(self):
        """Tensor(buffer, dtype) COPIES, so the token/tdf entries are rebuilt each step."""
        ins = []
        for i, s in enumerate(self.specs):
            vid = self.vid_of_spec[i]
            ck = self.chunk_of_in_vid.get(vid)
            if ck is not None:
                ins.append(Tensor(s.tensorType(), ck))
            elif vid in self.dead_chunks:
                ins.append(Tensor(s.tensorType(), self.dead_chunks[vid]))
            else:
                ins.append(Tensor(self.host_feed[i],
                                  xir.Layout("*C") if i < 2 else s.tensorType()))
        return ins

    def padfix_dev(self):
        """slot0 = slot1 on each buffer the NEXT step reads as past: 8 heads x 256 B."""
        for ck in self.pad_chunks:
            for h in range(NH):
                base = h * HEAD_STRIDE_B
                ck.copyFrom(base, ck, base + SLOT_B, SLOT_B)

    def step(self, token, past_len):
        f = self.host_feed
        f[0][0, 0] = np.float32(token)
        f[1][0, :] = np.asarray([1.0, past_len, 0.0], dtype=np.float32)
        ins = self.build_ins()
        outs = self.outs
        t0 = time.perf_counter()
        self.sess.forward(ins, outs)
        t1 = time.perf_counter()
        for t in outs:
            t.waitForReady(DT.timedelta(milliseconds=50000))
        t2 = time.perf_counter()
        if RESET:
            self.dev.reset(1)
        t3 = time.perf_counter()
        lt = outs[0]
        buf = np.zeros(list(lt.dtype().shape), dtype=np.float16)
        lt.read(buf, lt.offset(), lt.chunk().byte_size)
        t4 = time.perf_counter()
        nread = 0
        if not self.kv_on or PAD == "host":
            for oidx, pvid in self.pair.items():
                t = outs[oidx]
                t.read(f[pvid], t.offset(), t.chunk().byte_size)
                nread += 1
        if self.kv_on and KV == "copy":
            for ck_in, ck_out in self.copy_pairs:
                ck_in.copyFrom(0, ck_out, 0, KV_B)
        if not self.feed_dev:
            for pvid in self.pair.values():
                f[pvid][:, :, 0, :] = f[pvid][:, :, 1, :]
        elif PAD == "dev":
            self.padfix_dev()
        t5 = time.perf_counter()
        flat = buf.reshape(-1).astype(np.float32)
        if LOGDIR:
            np.save("%s/%s_step%d.npy" % (LOGDIR, CFG, past_len), flat)
        order = np.argpartition(flat, -2)[-2:]
        top2 = sorted(flat[order], reverse=True)
        nxt = int(np.argmax(flat))
        t6 = time.perf_counter()
        return nxt, top2, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                           t6 - t5, t6 - t0), float(flat.min()), float(flat.max())


def _safe_total(m):
    try:
        return m.sess.totalTime()
    except Exception as e:
        return "ERR %r" % (e,)


def dump_prof(m, tag):
    """Per-op composition of the runtime's own view of one forward.

    Returns nothing; writes <OUT stem>_ops_<tag>.csv next to the log.
    """
    try:
        rr = m.sess.timeProfileResults()
    except Exception as e:
        say("timeProfileResults FAILED %r" % (e,))
        return
    tot = mem = hard = oth = 0.0
    rows = []
    for op_id, v in rr.items():
        try:
            a, b, c, d = v
        except Exception:
            continue
        tot += a
        mem += b
        hard += c
        oth += d
        rows.append((b, a, c, d, op_id))
    say("PROFILE %s entries=%d total_time=%r" % (tag, len(rr), _safe_total(m)))
    say("PROFILE %s SUM total=%.1f memcpy=%.1f hard=%.1f other=%.1f ms "
        "=> memcpy %.1f %% hard %.1f %% other %.1f %%" % (
            tag, tot, mem, hard, oth,
            100.0 * mem / tot if tot else 0.0,
            100.0 * hard / tot if tot else 0.0,
            100.0 * oth / tot if tot else 0.0))
    rows.sort(reverse=True)
    path = OUT.replace(".txt", "_ops_%s.csv" % tag)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("op_id,memcpy_ms,total_ms,hard_ms,other_ms\n")
        for b, a, c, d, op_id in rows:
            fh.write("%s,%.6f,%.6f,%.6f,%.6f\n" % (op_id, b, a, c, d))
    say("PROFILE %s CSV %s rows=%d" % (tag, path, len(rows)))
    say("PROFILE %s TOP memcpy (op_id, memcpy_ms) %s" % (
        tag, [(int(i), round(b, 3)) for b, a, c, d, i in rows[:10]]))


def main():
    say("=== e41_kvres CFG=%s pid=%d kv=%s pad=%s dead=%s reset=%s steps=%d reg=%s align=%d" % (
        CFG, os.getpid(), KV, PAD, DEAD, RESET, STEPS, REG, ALIGN))
    say("FEED prefix=%r (%d) logdir=%s json=%s" % (PREFIX[:14], len(PREFIX), LOGDIR, JSON_))
    say("icraft=%r numpy=%r python=%r" % (getattr(icraft, "__version__", "?"),
                                          np.__version__, sys.version.split()[0]))
    say("MEM start %s | %s" % (read_stat(("VmRSS", "VmHWM")), meminfo()))
    wd = Watchdog(CMA_FLOOR)
    wd.start()
    m = E41()
    cols = [[] for _ in range(7)]
    chain, rows = [], []
    tok, past = 4, 1
    for i in range(STEPS):
        fed = PREFIX[i] if i < len(PREFIX) else tok
        tok, top2, d, lmin, lmax = m.step(fed, past)
        past += 1
        chain.append(tok)
        rows.append({"step": i, "fed": fed, "past_len": past - 1, "argmax": tok,
                     "logits_min": lmin, "logits_max": lmax,
                     "top1_minus_top2": round(float(top2[0] - top2[1]), 4)})
        for j, v in enumerate(d):
            cols[j].append(v)
        if i < 4 or i == STEPS - 1:
            say("step %02d past_len=%d fed=%d tok=%d top1=%.4f gap=%.4f fwd=%.3f "
                "wait=%.3f kv=%.3f all=%.3f (ms)" % (
                    i, past - 1, fed, tok, float(top2[0]), float(top2[0] - top2[1]),
                    d[0] * 1e3, d[1] * 1e3, d[4] * 1e3, d[6] * 1e3))
        if i % 5 == 0:
            say("  tick %d CmaFree=%d kB %s" % (i, cma_free(), read_stat(("VmRSS",))))
        if PROF and i == 1:
            dump_prof(m, "early")
    if PROF:
        dump_prof(m, "late")
    kvlab = {"off": "read_kv(host)", "bindonly": "read_kv(host)",
             "inplace": "padfix_dev", "copy": "kvcopy+padfix"}[KV]
    for nm, j in (("forward", 0), ("waitForReady", 1), ("reset", 2),
                  ("read_logits", 3), (kvlab, 4), ("argmax", 5), ("step_total", 6)):
        summarise(nm, cols[j])
    say("RESULT cfg=%s kv=%s pad=%s dead=%s reset=%s med_forward=%.2f med_step=%.2f" % (
        CFG, KV, PAD, DEAD, RESET, pct([v * 1e3 for v in cols[0]], 0.5),
        pct([v * 1e3 for v in cols[6]], 0.5)))
    say("CHAIN cfg=%s %s" % (CFG, chain))
    if OUT_JSON:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        say("JSON  %s (%d steps)" % (OUT_JSON, len(rows)))
    say("CMA low during run = %s kB" % wd.low)
    wd.stop = True
    say("MEM end %s | %s" % (read_stat(("VmRSS", "VmHWM")), meminfo()))
    say("=== done cfg=%s out=%s" % (CFG, OUT))


main()
