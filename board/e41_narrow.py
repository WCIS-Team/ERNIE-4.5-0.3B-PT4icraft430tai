#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41 narrow-read probe: does reading only the freshly written KV slot pay off?

Production path unchanged -- the load chain and step order are e41_chatd.py:135-160
(single-argument Session.forward), the same order e41_prof.py MODE=wall timed as the
six segments in ledger 28.1.  The only knob is how the 36 present tensors come back:

  E41N_READ=whole   36 reads x 1,048,576 B            (what the daemon does today)
  E41N_READ=narrow  288 reads x 256 B at slot=past_len

Narrow is only valid if present slots [0, past_len) are a pass-through of the past we
fed.  E41N_AUDIT=1 tests that directly: on the last step it does the narrow read AND a
whole read into a scratch buffer, then reports, per tensor, which 256 B fragments
differ and in which slots.  A bit-exact 12-step token chain is the second gate.

Nothing here writes model files, rebinds segments, or touches the daemon.
"""
import datetime as DT
import json
import os
import sys
import time

import numpy as np

sys.path.append("/root/qwen2/modelzoo_utils")
import icraft                       # noqa: E402
from icraft import xir, xrt         # noqa: E402
from icraft import zg330backend     # noqa: E402
from pyrtutils.icraft_utils import openDevice, initSession   # noqa: E402

Tensor = xrt.Tensor
JSON_ = os.environ.get("E41_JSON", "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.json")
RAW_ = os.environ.get("E41_RAW", "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.raw")
CFG = os.environ.get("E41N_CFG", "narrow")
READ = os.environ.get("E41N_READ", "narrow")
STEPS = int(os.environ.get("E41N_STEPS", "12"))
AUDIT = os.environ.get("E41N_AUDIT", "1") == "1"
OCM = os.environ.get("E41P_OCM", "1")
PADFIX = os.environ.get("E41_PADFIX", "1") == "1"
OUT = os.environ.get("E41N_OUT", "/mnt/data/ernie/narrow/%s.txt" % CFG)
OUT_JSON = os.environ.get("E41N_OUTJSON", "")

KV_BYTES = 1048576         # fp16[1,8,512,128]
HEADS = 8
DEPTH = 512
FRAG = 256                 # one head, one slot = 128 x fp16
N_KV = 36
TOTAL_KV = KV_BYTES * N_KV


def use_graph_geom(kv_shape, itemsize, n_kv):
    """Rebind the cache geometry to what the compiled network actually declares.

    NS comes from the toml's max_tokens_length at compile time, so a shorter-cache
    件 has a different NS and every buffer this file sizes must follow the graph.
    On E41 (NS=512) the derived values equal the constants above: a no-op there.
    """
    global KV_BYTES, HEADS, DEPTH, FRAG, N_KV, TOTAL_KV
    HEADS, DEPTH = int(kv_shape[1]), int(kv_shape[2])
    FRAG = int(kv_shape[3]) * itemsize
    KV_BYTES = HEADS * DEPTH * FRAG
    N_KV = n_kv
    TOTAL_KV = KV_BYTES * N_KV
    say("GEOM kv_shape=%r itemsize=%d n_kv=%d -> HEADS=%d DEPTH=%d FRAG=%d "
        "KV_BYTES=%d TOTAL_KV=%d" % (kv_shape, itemsize, n_kv, HEADS, DEPTH, FRAG,
                                     KV_BYTES, TOTAL_KV))


_LOG = None


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    if _LOG is not None:
        _LOG.write(line + "\n")
        _LOG.flush()


def meminfo(keys=("CmaFree", "MemAvailable")):
    out = {}
    try:
        for ln in open("/proc/meminfo"):
            k = ln.split(":")[0].strip()
            if k in keys:
                out[k] = int(ln.split()[1])
    except Exception:
        pass
    return " ".join("%s=%d kB" % (k, out[k]) for k in keys if k in out)


def read_stat(keys=("VmRSS", "VmHWM")):
    want = ("VmSize", "VmRSS", "VmHWM", "VmSwap")
    out = {}
    for ln in open("/proc/self/status"):
        parts = ln.split()
        if parts and parts[0].rstrip(":") in want:
            out[parts[0].rstrip(":")] = int(parts[1])
    return " ".join("%s=%d kB" % (k, out.get(k, -1)) for k in keys)


def load_graph_map(jsonpath):
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
    return {"pair": pair, "dead": dead,
            "n_past": len([v for v in [o["v_id"] for o in ins]
                           if v > 1 and v not in dead])}


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


class E41(object):
    def __init__(self):
        t0 = time.time()
        self.net = xir.Network.CreateFromJsonFile(JSON_)
        self.net.lazyLoadParamsFromFile(RAW_)
        say("json+raw %.1f s" % (time.time() - t0))
        self.gmap = load_graph_map(JSON_)
        say("pair=%d n_past=%d dead=%s" % (len(self.gmap["pair"]),
                                           self.gmap["n_past"], self.gmap["dead"]))
        if len(self.gmap["pair"]) != self.gmap["n_past"]:
            raise SystemExit("KV pairing incomplete")
        self.specs = list(self.net.ops[0].outputs)
        if len(self.specs) != 42:
            raise SystemExit("expected 42 declared inputs, got %d" % len(self.specs))
        self.feed = []
        for s in self.specs:
            t = s.tensorType()
            dt = np.float32 if "fp(32)" in str(t.element_dtype) else np.float16
            self.feed.append(np.zeros(list(t.shape), dtype=dt))
        # uint8 alias of every KV past buffer: [head, slot, FRAG B], same memory
        pvid0 = sorted(set(self.gmap["pair"].values()))[0]
        t0_ = self.specs[pvid0].tensorType()
        use_graph_geom(list(t0_.shape), np.dtype(np.float16).itemsize,
                       len(self.gmap["pair"]))
        self.u8 = {}
        for pvid in set(self.gmap["pair"].values()):
            self.u8[pvid] = self.feed[pvid].view(np.uint8).reshape(HEADS, DEPTH, FRAG)
        self.logits = np.zeros(list(self.net.ops[-1].inputs[0].tensorType().shape),
                               dtype=np.float16)
        self.dev = openDevice("zg330", "192.168.125.143", True, False)
        self.sess = initSession("zg330", self.net, self.dev, int(OCM))
        self.zg = zg330backend.ZG330Backend(self.sess.backends[0])
        t0 = time.time()
        self.sess.apply()
        say("apply %.1f s" % (time.time() - t0))
        say("CAP ocmopt=%r is_etmOptimize=%r is_mergeHardop=%r is_applied=%r" % (
            getattr(self.zg, "ocmopt", "?"), getattr(self.zg, "is_etmOptimize", "?"),
            getattr(self.zg, "is_mergeHardop", "?"), getattr(self.zg, "is_applied", "?")))
        self.narrow_calls = 0

    def tensors(self):
        return [Tensor(a, xir.Layout("*C") if i < 2 else s.tensorType())
                for i, (a, s) in enumerate(zip(self.feed, self.specs))]

    def read_kv_whole(self, outs):
        n = 0
        for oidx, pvid in self.gmap["pair"].items():
            buf = self.feed[pvid]
            t = outs[oidx]
            t.read(buf, t.offset(), t.chunk().byte_size)
            if PADFIX:
                buf[:, :, 0, :] = buf[:, :, 1, :]
            n += 1
        return n

    def read_kv_narrow(self, outs, slot):
        if slot >= DEPTH:
            raise SystemExit("slot %d out of range" % slot)
        n = 0
        for oidx, pvid in self.gmap["pair"].items():
            t = outs[oidx]
            base = t.offset()
            dst = self.u8[pvid]
            for h in range(HEADS):
                t.read(dst[h, slot], base + (h * DEPTH + slot) * FRAG, FRAG)
                self.narrow_calls += 1
            if PADFIX:
                self.feed[pvid][:, :, 0, :] = self.feed[pvid][:, :, 1, :]
            n += 1
        return n

    def audit(self, outs, slot):
        """Narrow-filled cache vs a fresh whole read of the same present tensors."""
        diffs = []
        for oidx, pvid in self.gmap["pair"].items():
            t = outs[oidx]
            ref = np.zeros(list(t.dtype().shape), dtype=np.float16)
            t.read(ref, t.offset(), t.chunk().byte_size)
            ours = self.feed[pvid]
            if PADFIX:
                ref[:, :, 0, :] = ref[:, :, 1, :]
            a = ours.view(np.uint8).reshape(HEADS, DEPTH, FRAG)
            b = ref.view(np.uint8).reshape(HEADS, DEPTH, FRAG)
            ne = int((a != b).any(axis=2).sum())
            if ne:
                slots = sorted({int(s) for s in np.nonzero((a != b).any(axis=2))[1]})
                diffs.append({"pvid": pvid, "frag_diff": ne, "slots": slots[:12],
                              "n_slots": len(slots)})
        say("AUDIT slot=%d tensors_with_diff=%d/%d identical=%s" % (
            slot, len(diffs), N_KV, "yes" if not diffs else "NO"))
        for d in diffs[:6]:
            say("AUDIT  diff v%d frags=%d n_slots=%d slots=%s" % (
                d["pvid"], d["frag_diff"], d["n_slots"], d["slots"]))
        return {"slot": slot, "diff_tensors": len(diffs),
                "detail": diffs, "clean": not diffs}

    def step(self, token, past_len):
        f = self.feed
        f[0][0, 0] = np.float32(token)
        f[1][0, :] = np.asarray([1.0, past_len, 0.0], dtype=np.float32)
        t0 = time.perf_counter()
        outs = self.sess.forward(self.tensors())
        self.last_outs = outs
        t1 = time.perf_counter()
        for t in outs:
            t.waitForReady(DT.timedelta(milliseconds=50000))
        t2 = time.perf_counter()
        self.dev.reset(1)
        t3 = time.perf_counter()
        ch = outs[0].chunk()
        outs[0].read(self.logits, outs[0].offset(), ch.byte_size)
        t4 = time.perf_counter()
        if READ == "narrow":
            n = self.read_kv_narrow(outs, past_len)
        else:
            n = self.read_kv_whole(outs)
        t5 = time.perf_counter()
        flat = self.logits.reshape(-1).astype(np.float32)
        top2 = sorted(np.argpartition(flat, -2)[-2:], key=lambda i: -flat[i])
        nxt = int(np.argmax(flat))
        t6 = time.perf_counter()
        return (nxt, float(flat[top2[0]]), float(flat[top2[0]] - flat[top2[1]]), n,
                (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0))


def main():
    global _LOG
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    _LOG = open(OUT, "w", encoding="utf-8")
    say("=== e41_narrow CFG=%s pid=%d read=%s steps=%d audit=%s padfix=%s ocm=%s" % (
        CFG, os.getpid(), READ, STEPS, AUDIT, PADFIX, OCM))
    say("icraft=%r numpy=%r python=%r" % (getattr(icraft, "__version__", "?"),
                                          np.__version__, sys.version.split()[0]))
    say("MEM start %s | %s" % (read_stat(), meminfo()))
    m = E41()
    say("MEM after apply %s | %s" % (read_stat(), meminfo()))
    cols = [[] for _ in range(7)]
    chain, gaps, rows = [], [], []
    tok, past = 4, 1
    aud = None
    for i in range(STEPS):
        nxt, top1, gap, nread, d = m.step(tok, past)
        tok = nxt
        past += 1
        chain.append(tok)
        gaps.append(round(gap, 4))
        rows.append({"step": i, "past_len": past - 1, "argmax": tok, "top1": top1,
                     "top1_minus_top2": round(gap, 4),
                     "reads": nread, "bytes": nread * FRAG if READ == "narrow"
                     else nread * KV_BYTES,
                     "ms": [round(x * 1e3, 3) for x in d]})
        for j, v in enumerate(d):
            cols[j].append(v)
        if i < 4 or i == STEPS - 1:
            say("step %02d past_len=%d tok=%d top1=%.4f gap=%.4f fwd=%.3f kv=%.3f "
                "all=%.3f (ms)" % (i, past - 1, tok, top1, gap,
                                   d[0] * 1e3, d[4] * 1e3, d[6] * 1e3))
        if AUDIT and i == STEPS - 1:
            aud = m.audit(m.last_outs, past - 1)
    say("READ calls_per_step=%d bytes_per_step=%d" % (
        m.narrow_calls // max(STEPS, 1),
        N_KV * FRAG * HEADS if READ == "narrow" else N_KV * KV_BYTES))
    lab = "read_kv(narrow)" if READ == "narrow" else "read_kv(whole)"
    for nm, j in (("forward", 0), ("waitForReady", 1), ("reset", 2),
                  ("read_logits", 3), (lab, 4), ("argmax", 5), ("step_total", 6)):
        summarise(nm, cols[j])
    tail = [cols[3][i] + cols[4][i] + cols[5][i] for i in range(STEPS)]
    say("PRODUCTION_DT_EQUIV med=%.3f ms (forward+wait+reset)" % (
        pct([(cols[0][i] + cols[1][i] + cols[2][i]) * 1000 for i in range(STEPS)], 0.5)))
    say("POST_DT_TAIL med=%.3f ms (read_logits+kv+argmax)" % (
        pct([v * 1000 for v in tail], 0.5)))
    say("RESULT cfg=%s read=%s med_forward=%.2f med_kv=%.2f med_step=%.2f "
        "tokens_per_s=%.3f" % (
            CFG, READ, pct([v * 1e3 for v in cols[0]], 0.5),
            pct([v * 1e3 for v in cols[4]], 0.5),
            pct([v * 1e3 for v in cols[6]], 0.5),
            1000.0 / max(pct([v * 1e3 for v in cols[6]], 0.5), 1e-9)))
    say("CHAIN cfg=%s %s" % (CFG, chain))
    say("GAPS  cfg=%s %s" % (CFG, gaps))
    if OUT_JSON:
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump({"cfg": CFG, "read": READ, "chain": chain, "gaps": gaps,
                       "audit": aud, "rows": rows}, fh, ensure_ascii=False)
        say("JSON  %s (%d steps)" % (OUT_JSON, len(rows)))
    say("MEM end %s | %s" % (read_stat(), meminfo()))
    say("=== done cfg=%s out=%s" % (CFG, OUT))


main()
