#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41 ocm/etm/merge single-variable sweep on the board (item 2 + item 3).

One config per process: initSession() is the only thing that changes, so a
session can never be re-tuned in place -- ocmOptimize/disable* must run before
apply(), which means a fresh process and a fresh apply() for every rung.

  E41P_OCM   0/1/2/3 = OcmOpt.NONE/OPTION1/OPTION2/OPTION3 via the helper
             4       = helper does not call ocmOptimize at all (its sentinel)
  E41P_BEST  1      = call ZG330Backend.ocmOptimize(BEST_SCORE) by hand (needs
                      E41P_OCM=4, the Python helper cannot express -1)
  E41P_ETM   off    = disableEtmOptimize()   (helper open_compressFtmp=False)
  E41P_MERGE off    = disableMergeHardOp()   (helper open_speedmode=False)
  E41P_PRECHECK 1  = also run precheck(); costs ~43 s on this board, off by default

Readings: json+raw / mid / apply seconds, apply-time segment sizes (logic+phy),
CAP readback (which option the runtime actually settled on), a greedy token
chain with top1-top2 gaps (numeric fingerprint), per-step segment timings, CMA.

Nothing here writes or deletes model files, and nothing touches the daemon.
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
CFG = os.environ.get("E41P_CFG", "ocm1")
OCM_OPT = int(os.environ.get("E41P_OCM", "1"))
BEST = os.environ.get("E41P_BEST", "0") == "1"
ETM = os.environ.get("E41P_ETM", "on")
MERGE = os.environ.get("E41P_MERGE", "on")
PRECHK = os.environ.get("E41P_PRECHECK", "0") == "1"
STEPS = int(os.environ.get("E41P_STEPS", "12"))
PADFIX = os.environ.get("E41_PADFIX", "1") == "1"
OUT = os.environ.get("E41P_OUT", "/mnt/data/ernie/ocm/%s.txt" % CFG)
# 喂一段真提示（逗号分隔 id），不设就按老行为从 token 4 空跑；喂完转回贪心接续
PREFIX = [int(x) for x in os.environ.get("E41P_PREFIX", "").replace(" ", "").split(",") if x]
OUT_JSON = os.environ.get("E41P_OUTJSON", "")

SEG = {0: "WEIGHT", 1: "INSTR", 2: "FTMP", 3: "INPUT", 4: "OUTPUT"}
_LOG = open(OUT, "w", encoding="utf-8")


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    _LOG.write(line + "\n")
    _LOG.flush()


def read_stat(keys, path="/proc/self/status"):
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
    v = [x * 1000.0 for x in vals]
    say("SEG %-16s n=%d min=%.2f med=%.2f p95=%.2f max=%.2f ms" % (
        name, len(v), min(v), pct(v, 0.5), pct(v, 0.95), max(v)))


def num(v):
    for attr in ("__int__",):
        if hasattr(v, attr):
            try:
                return int(v)
            except Exception:
                pass
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
            parts.append("%s=?" % SEG.get(int(st), st))
            continue
        tot += b
        parts.append("%s=%d" % (SEG.get(int(st), st), b))
    say("SEGMAP %s %s TOTAL=%d B (%.1f MiB)" % (
        label, " ".join(parts), tot, tot / 1048576.0))


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
        t_mid = time.time()
        self.dev = openDevice("zg330", "192.168.125.143", True, False)
        self.sess = initSession("zg330", self.net, self.dev, OCM_OPT,
                                mmuMode=True, open_speedmode=(MERGE == "on"),
                                open_compressFtmp=(ETM == "on"))
        self.zg = zg330backend.ZG330Backend(self.sess.backends[0])
        if BEST:
            opt = getattr(zg330backend.OcmOpt, "BEST_SCORE", None)
            if opt is None:
                raise SystemExit("OcmOpt.BEST_SCORE missing")
            self.zg.ocmOptimize(opt)
            say("manual ocmOptimize(BEST_SCORE=%r) ok" % (int(opt),))
        say("mid %.1f s (graph map+openDevice+initSession)" % (time.time() - t_mid))
        dump_seg(self.zg, "logic_segment_map", "logic(before apply)")
        t0 = time.time()
        try:
            self.sess.apply()
        except Exception as e:
            say("APPLY FAILED after %.1f s: %r" % (time.time() - t0, e))
            raise
        say("apply %.1f s" % (time.time() - t0))
        say("CAP ocmopt=%r is_etmOptimize=%r is_mergeHardop=%r is_applied=%r" % (
            getattr(self.zg, "ocmopt", "?"), getattr(self.zg, "is_etmOptimize", "?"),
            getattr(self.zg, "is_mergeHardop", "?"), getattr(self.zg, "is_applied", "?")))
        dump_seg(self.zg, "phy_segment_map", "phy(after apply)")
        if PRECHK:
            t0 = time.time()
            try:
                say("precheck=%r (%.1f s)" % (self.zg.precheck(), time.time() - t0))
            except Exception as e:
                say("precheck ERR %r (%.1f s)" % (e, time.time() - t0))

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
        flat = self.logits.reshape(-1).astype(np.float32)
        order = np.argpartition(flat, -2)[-2:]
        pair2 = sorted(flat[order], reverse=True)
        nxt = int(np.argmax(flat))
        t6 = time.perf_counter()
        return nxt, pair2, outs, (t1 - t0, t2 - t1, t3 - t2, t4 - t3,
                                  t5 - t4, t6 - t5, t6 - t0), \
            float(flat.min()), float(flat.max())


def main():
    say("=== e41_ocm CFG=%s pid=%d ocm=%d best=%s etm=%s merge=%s steps=%d "
        "padfix=%s precheck=%s" % (CFG, os.getpid(), OCM_OPT, BEST, ETM, MERGE,
                                   STEPS, PADFIX, PRECHK))
    say("icraft=%r numpy=%r python=%r" % (getattr(icraft, "__version__", "?"),
                                          np.__version__, sys.version.split()[0]))
    say("MEM start %s | %s" % (read_stat(("VmRSS", "VmHWM")), meminfo()))
    m = E41()
    cols = [[] for _ in range(7)]
    chain, gaps, rows = [], [], []
    tok, past = 4, 1
    for i in range(STEPS):
        fed = PREFIX[i] if i < len(PREFIX) else tok
        tok, top2, outs, d, lmin, lmax = m.step(fed, past)
        past += 1
        chain.append(tok)
        gaps.append(round(float(top2[0] - top2[1]), 4))
        rows.append({"step": i, "fed": fed, "past_len": past - 1, "argmax": tok,
                     "logits_min": lmin, "logits_max": lmax,
                     "top1_minus_top2": round(float(top2[0] - top2[1]), 4)})
        for j, v in enumerate(d):
            cols[j].append(v)
        if i < 3 or i == STEPS - 1:
            say("step %02d past_len=%d tok=%d top1=%.4f gap=%.4f fwd=%.3f "
                "wait=%.3f kvread=%.3f all=%.3f (ms)" % (
                    i, past - 1, tok, float(top2[0]), float(top2[0] - top2[1]),
                    d[0] * 1e3, d[1] * 1e3, d[4] * 1e3, d[6] * 1e3))
    for nm, j in (("forward", 0), ("waitForReady", 1), ("reset", 2),
                  ("read_logits", 3), ("read_kv36", 4), ("argmax", 5),
                  ("step_total", 6)):
        summarise(nm, cols[j])
    dt_equiv = [cols[0][i] + cols[1][i] + cols[2][i] for i in range(STEPS)]
    tail = [cols[3][i] + cols[4][i] + cols[5][i] for i in range(STEPS)]
    say("RESULT cfg=%s apply_med_forward=%.2f wall_step=%.2f dt_equiv=%.2f "
        "post_dt_tail=%.2f" % (CFG, pct([v * 1e3 for v in cols[0]], 0.5),
                               pct([v * 1e3 for v in cols[6]], 0.5),
                               pct([v * 1e3 for v in dt_equiv], 0.5),
                               pct([v * 1e3 for v in tail], 0.5)))
    say("CHAIN cfg=%s %s" % (CFG, chain))
    say("GAPS  cfg=%s %s" % (CFG, gaps))
    if OUT_JSON:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        say("JSON  %s (%d 步)" % (OUT_JSON, len(rows)))
    say("MEM end %s | %s" % (read_stat(("VmRSS", "VmHWM")), meminfo()))
    say("=== done cfg=%s out=%s" % (CFG, OUT))


main()
