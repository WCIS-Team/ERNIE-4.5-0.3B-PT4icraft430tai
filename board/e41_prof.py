#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41 (ERNIE-4.5-0.3B ZG) board-side timing probe.

E41P_MODE=wall : runtime profiling OFF; measure our own per-step segments and the
                 Tensor.read cost curve (item A-wall + item B).
E41P_MODE=prof : Session.enableTimeProfile(True) BEFORE apply, then dump
                 timeProfileResults() (item A-prof; official docs say profiling
                 inflates software time, so these numbers are composition only).

Nothing here writes model files or changes the production daemon.
"""
import datetime as DT
import json
import os
import sys
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
MODE = os.environ.get("E41P_MODE", "wall")
STEPS = int(os.environ.get("E41P_STEPS", "24"))
REPS = int(os.environ.get("E41P_REPS", "5"))
OCM = os.environ.get("E41P_OCM", "1")
PADFIX = os.environ.get("E41_PADFIX", "1") == "1"
SLOT = int(os.environ.get("E41P_SLOT", "8"))
OUT = os.environ.get("E41P_OUT", "/tmp/e41p_%s.txt" % MODE)

KV_BYTES = 1048576          # fp16[1,8,512,128]
HEADS = 8
DEPTH = 512
HDIM = 128
FRAG = HDIM * 2             # 256 B = one head, one slot
N_KV = 36
TOTAL_KV = KV_BYTES * N_KV  # 37,748,736 B

_LOG = open(OUT, "w", encoding="utf-8")


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    _LOG.write(line + "\n")
    _LOG.flush()


def read_stat(keys, path="/proc/self/status"):
    want = ("VmSize", "VmRSS", "VmHWM", "VmSwap")
    out = {}
    for ln in open(path):
        k = ln.split(":")[0]
        if k in keys:
            out[k] = ln.split()[1] + ln.split()[2]
    return " ".join("%s=%s" % (w, out.get(w, "?")) for w in keys)


def meminfo():
    want = ("MemAvailable", "CmaFree", "Cached")
    out = {}
    for ln in open("/proc/meminfo"):
        k = ln.split(":")[0]
        if k in want:
            out[k] = ln.split()[1]
    return " ".join("%s=%s kB" % (k, out.get(k, "?")) for k in want)


def load_graph_map(jsonpath):
    """present out-index -> the past input v_id it was sliced from."""
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
            "n_past": len([v for v in [o["v_id"] for o in ins]
                           if v > 1 and v not in dead])}


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def summarise(name, vals):
    vals_ms = [v * 1000.0 for v in vals]
    say("SEG %-16s n=%d min=%.2f med=%.2f p95=%.2f max=%.2f ms" % (
        name, len(vals_ms), min(vals_ms), pct(vals_ms, 0.5),
        pct(vals_ms, 0.95), max(vals_ms)))


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
        self.feed = []
        for s in self.specs:
            t = s.tensorType()
            dt = np.float32 if "fp(32)" in str(t.element_dtype) else np.float16
            self.feed.append(np.zeros(list(t.shape), dtype=dt))
        self.logits = np.zeros(list(self.net.ops[-1].inputs[0].tensorType().shape),
                               dtype=np.float16)
        self.dev = openDevice("zg330", "192.168.125.143", True, False)
        self.sess = initSession("zg330", self.net, self.dev, int(OCM))
        self.zg = zg330backend.ZG330Backend(self.sess.backends[0])
        if MODE == "prof":
            try:
                self.sess.enableTimeProfile(True)
                say("enableTimeProfile(True) ok, time_profile=%r" % (
                    getattr(self.sess, "time_profile", "?"),))
            except Exception as e:
                say("enableTimeProfile FAILED %r" % (e,))
        t0 = time.time()
        self.sess.apply()
        say("apply %.1f s" % (time.time() - t0))
        say("CAP ocmopt=%r is_etmOptimize=%r is_mergeHardop=%r is_applied=%r" % (
            getattr(self.zg, "ocmopt", "?"), getattr(self.zg, "is_etmOptimize", "?"),
            getattr(self.zg, "is_mergeHardop", "?"), getattr(self.zg, "is_applied", "?")))
        try:
            say("precheck=%r" % (self.zg.precheck(),))
        except Exception as e:
            say("precheck ERR %r" % (e,))

    def tensors(self):
        return [Tensor(a, xir.Layout("*C") if i < 2 else s.tensorType())
                for i, (a, s) in enumerate(zip(self.feed, self.specs))]

    def step(self, token, past_len):
        f = self.feed
        f[0][0, 0] = np.float32(token)
        f[1][0, :] = np.asarray([1.0, past_len, 0.0], dtype=np.float32)
        t0 = time.perf_counter()
        outs = self.sess.forward(self.tensors())
        t1 = time.perf_counter()
        for t in outs:
            t.waitForReady(DT.timedelta(milliseconds=50000))
        t2 = time.perf_counter()
        self.dev.reset(1)
        t3 = time.perf_counter()
        ch = outs[0].chunk()
        outs[0].read(self.logits, outs[0].offset(), ch.byte_size)
        t4 = time.perf_counter()
        for oidx, pvid in self.gmap["pair"].items():
            buf = f[pvid]
            t = outs[oidx]
            t.read(buf, t.offset(), t.chunk().byte_size)
            if PADFIX:
                buf[:, :, 0, :] = buf[:, :, 1, :]
        t5 = time.perf_counter()
        nxt = int(np.argmax(self.logits.reshape(-1)))
        t6 = time.perf_counter()
        return nxt, outs, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0)


def run_steps(m):
    cols = [[] for _ in range(7)]
    tok, outs, past = 4, None, 1
    for i in range(STEPS):
        tok, outs, d = m.step(tok, past)
        past += 1
        for j, v in enumerate(d):
            cols[j].append(v)
        if i < 3 or i == STEPS - 1:
            say("step %02d past_len=%d fwd=%.3f wait=%.3f reset=%.3f "
                "logits=%.3f kvread=%.3f argmax=%.3f all=%.3f (ms)" % (
                    i, past - 1, d[0] * 1e3, d[1] * 1e3, d[2] * 1e3,
                    d[3] * 1e3, d[4] * 1e3, d[5] * 1e3, d[6] * 1e3))
    for nm, j in (("forward", 0), ("waitForReady", 1), ("reset", 2),
                  ("read_logits", 3), ("read_kv36", 4), ("argmax", 5),
                  ("step_total", 6)):
        summarise(nm, cols[j])
    dt_equiv = [cols[0][i] + cols[1][i] + cols[2][i] for i in range(STEPS)]
    say("PRODUCTION_DT_EQUIV med=%.3f ms (forward+wait+reset == e41_chatd dt)" % (
        pct([v * 1000 for v in dt_equiv], 0.5)))
    say("WALL_STEP med=%.3f ms (== round-level s/step)" % (
        pct([c * 1000 for c in cols[6]], 0.5)))
    tail = [cols[3][i] + cols[4][i] + cols[5][i] for i in range(STEPS)]
    say("POST_DT_TAIL med=%.3f ms (read_logits+read_kv36+argmax == the ceiling "
        "for narrow read)" % (pct([v * 1000 for v in tail], 0.5)))
    say("READ_KV36 share of WALL_STEP = %.1f %%" % (
        100.0 * pct(cols[4], 0.5) / max(pct(cols[6], 0.5), 1e-9)))
    return outs


def battery(m, outs):
    kv = [(o, outs[o]) for o in sorted(m.gmap["pair"])]
    if len(kv) != N_KV:
        say("BATTERY skip: kv outs=%d != %d" % (len(kv), N_KV))
        return
    d1 = np.zeros(KV_BYTES, dtype=np.uint8)
    df = np.zeros(FRAG, dtype=np.uint8)
    d2 = np.zeros(2048, dtype=np.uint8)
    src = np.zeros(KV_BYTES, dtype=np.uint8)

    def full():
        for _, t in kv:
            t.read(d1, t.offset(), t.chunk().byte_size)

    def frag():
        for _, t in kv:
            base = t.offset()
            for h in range(HEADS):
                t.read(df, base + (h * DEPTH + SLOT) * FRAG, FRAG)

    def row2k():
        for _, t in kv:
            t.read(d2, t.offset() + SLOT * FRAG, 2048)

    def host():
        for _ in range(N_KV):
            np.copyto(d1, src)

    res = {}
    for nm, fn, calls, nbytes in (("read_36x1MiB", full, 36, TOTAL_KV),
                                  ("read_288x256B", frag, 288, 288 * FRAG),
                                  ("read_36x2KiB", row2k, 36, 36 * 2048),
                                  ("host_memcpy36MiB", host, 36, TOTAL_KV)):
        ts = []
        for _ in range(REPS):
            a = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - a)
        res[nm] = (min(ts), pct(ts, 0.5), calls, nbytes)
        say("BATT %-20s calls=%d bytes=%d min=%.2f med=%.2f ms -> %.1f MB/s" % (
            nm, calls, nbytes, min(ts) * 1e3, pct(ts, 0.5) * 1e3,
            nbytes / max(min(ts), 1e-9) / 1e6))
    # two-point solve of T = a*calls + r*bytes using full vs frag
    tf, bf, cf = res["read_36x1MiB"][0], res["read_36x1MiB"][3], 36
    tr_, br_, cr = res["read_288x256B"][0], res["read_288x256B"][3], 288
    det = cf * br_ - cr * bf
    if det:
        a = (tf * br_ - tr_ * bf) / det
        r = (cf * tr_ - cr * tf) / det
        say("SOLVE per-call a=%.1f us  per-byte r=%.1f MB/s (%.3f ns/B)" % (
            a * 1e6, 1 / r / 1e6 if r else float("nan"), r * 1e9))
        pred_frag = 288 * a + br_ * r
        say("PRED frag=%.2f ms vs measured %.2f ms; net saving if narrow: %.2f ms/step" % (
            pred_frag * 1e3, tr_ * 1e3, (tf - pred_frag) * 1e3))
        say("BREAKEVEN per-call overhead where narrow read stops paying = %.1f us "
            "(= full-read time / 288)" % (tf / 288 * 1e6,))
        chk = 36 * a + 36 * 2048 * r
        say("CHECK 36x2KiB predicted %.2f ms vs measured %.2f ms" % (
            chk * 1e3, res["read_36x2KiB"][0] * 1e3))
    say("HOST baseline %.2f ms for the same %d B" % (
        res["host_memcpy36MiB"][0] * 1e3, TOTAL_KV))


def dump_profile(m, tag):
    try:
        rr = m.sess.timeProfileResults()
    except Exception as e:
        say("timeProfileResults FAILED %r" % (e,))
        return None
    say("PROFILE %s entries=%d totalTime=%r" % (tag, len(rr),
                                                _safe_total(m)))
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
        rows.append((b, a, op_id))
    say("PROFILE %s SUM total=%.1f memcpy=%.1f hard=%.1f other=%.1f ms "
        "=> memcpy share %.1f %% hard share %.1f %%" % (
            tag, tot, mem, hard, oth,
            100.0 * mem / tot if tot else 0.0,
            100.0 * hard / tot if tot else 0.0))
    rows.sort(reverse=True)
    with open(OUT.replace(".txt", "_ops_%s.csv" % tag), "w", encoding="utf-8") as fh:
        fh.write("op_id,memcpy_ms,total_ms\n")
        for b, a, op_id in rows:
            fh.write("%s,%.6f,%.6f\n" % (op_id, b, a))
    top = [(int(i), round(b, 3)) for b, a, i in rows[:10]]
    say("TOP memcpy ops (op_id, memcpy_ms): %s" % (top,))
    return tot


def _safe_total(m):
    try:
        return m.sess.totalTime()
    except Exception as e:
        return "ERR %r" % (e,)


def main():
    say("=== e41_prof MODE=%s pid=%d steps=%d reps=%d ocm=%s padfix=%s slot=%d" % (
        MODE, os.getpid(), STEPS, REPS, OCM, PADFIX, SLOT))
    say("icraft=%r numpy=%r python=%r" % (getattr(icraft, "__version__", "?"),
                                          np.__version__, sys.version.split()[0]))
    say("MEM start %s | %s" % (read_stat(("VmRSS", "VmHWM")), meminfo()))
    m = E41()
    outs = run_steps(m)
    if MODE == "wall":
        battery(m, outs)
    else:
        dump_profile(m, "after_first_dump")
        for i in range(6):
            t0 = time.perf_counter()
            m.step(7, 3 + i)
            say("prof-step %d wall %.3f ms" % (i, (time.perf_counter() - t0) * 1e3))
        dump_profile(m, "after_7_steps")
    say("MEM end %s | %s" % (read_stat(("VmRSS", "VmHWM")), meminfo()))
    say("=== done out=%s" % OUT)


main()
