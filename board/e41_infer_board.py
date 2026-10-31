"""ERNIE-4.5-0.3B (E41 ZG) decode-only driver for the ZG330 board.

Same contract as the Host-backend reference `compile/host_gen.py`, which is the file that
produced the numbers this driver is checked against; the Qwen3 driver
(`Qwen3-0.6B-Instruct/patch/q3_infer_board.py`) is only the I/O template, its cache handling is
different (it compacts one slot per step, ERNIE chains the whole 512-slot present back).

Feed contract, read off E41-decoder_ZG.json (42 declared input values, 37 output values):
  v_id 0      fp32 [1,1]        the token of this step
  v_id 1      fp32 [1,3]        [1.0, past_len, 0.0]; element 1 is both write slot and RoPE
                                position, element 2 is read by nothing; past_len must be >= 1
                                or op 4525 (Reshape) sizes a 512x tensor and fails
  v_id 2..37  fp16 [1,8,512,128]  36 real KV inputs (18 layers x K,V)
  v_id 38..41 fp16 [1,8,512,128]  declared, consumed by nobody, still must be fed zero
                                  (runtime checks inputs size before dispatching)
Outputs: logits fp16 [1,1,103424], then 36 present tensors that go straight back as the next
step's past (whole tensor, no compaction -- host_gen does the same).

--padfix (env E41_PADFIX, default 1; set 0 to disable) repairs the pad slot: the graph never
writes slot 0, so position 0 attends to an all-zero pad.  The repair feeds token 0 twice and
copies slot 1 into slot 0 of every chained cache, so the repeat's window is {self, self}.
Default is 1 because the board A/B on 2026-11-01 showed padfix=0 degenerates into a timestamp
loop on the same prompt and padfix=1 answers coherently, at no measurable speed cost.

Run (board):
  E41_MAX_NEW=1 python3 e41_infer_board.py
  E41_TEXT='你好' E41_MAX_NEW=16 python3 e41_infer_board.py
  E41_PADFIX=0 E41_MAX_NEW=16 python3 e41_infer_board.py    # A/B off
"""
import datetime
import json
import math
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

import icraft
from icraft import xir

sys.path.append("/root/qwen2/modelzoo_utils")
from pyrtutils.icraft_utils import openDevice, initSession
from icraft import xrt

Tensor = xrt.Tensor

JSON_ = os.environ.get("E41_JSON", "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.json")
RAW_ = os.environ.get("E41_RAW", "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.raw")
CONFIG_DIR = os.environ.get("E41_CONFIG", "/mnt/data/ernie/config/ernie")
PROMPT = os.environ.get("E41_PROMPT", "94515,94036,31282,5502,44284,3458,93977")
TEXT = os.environ.get("E41_TEXT", "")
MAX_NEW = int(os.environ.get("E41_MAX_NEW", "8"))
PADFIX = os.environ.get("E41_PADFIX", "1") == "1"
STOP = {int(t) for t in os.environ.get("E41_STOP", "2").split(",") if t.strip()}
POS_OFFSET = int(os.environ.get("E41_POS_OFFSET", "1"))
MAX_SLOTS = 512


def mem(tag):
    st = {}
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in ("VmRSS", "VmHWM", "VmSwap"):
            st[k] = line.split()[1] + " " + line.split()[2]
    cm = {}
    for line in open("/proc/meminfo"):
        k = line.split(":")[0]
        if k in ("MemAvailable", "MemFree", "CmaFree"):
            cm[k] = line.split()[1]
    print("[MEM] %-14s %s | %s" % (tag, " ".join("%s=%s" % kv for kv in st.items()),
                                   " ".join("%s=%s" % kv for kv in cm.items())), flush=True)


def load_graph_map(jsonpath):
    """pair: present v_id -> the past v_id it was built from (walk back to SliceDynamic);
    dead: declared cache inputs with no consumer."""
    d = json.load(open(jsonpath, encoding="utf-8"))
    ops = d["ops"]
    producer = {}
    for op in ops:
        for o in op.get("outputs") or []:
            producer[o["v_id"]] = op
    ins = sorted(ops[0]["outputs"], key=lambda o: o["v_id"])
    outs = ops[-1]["inputs"]
    consumed = {i["v_id"] for op in ops[1:] for i in (op.get("inputs") or [])}
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
            "n_past": len([v for v in [o["v_id"] for o in ins] if v > 1 and v not in dead])}


class E41:
    def __init__(self):
        # board has tokenizers 0.13.3; ERNIE's tokenizer.json uses the newer BPE field
        # `ignore_merges`, which 0.13.3 refuses to deserialize. Tokenisation is therefore done
        # off-board and this driver takes ids; decode is best-effort only.
        try:
            self.tokenizer = Tokenizer.from_file(os.path.join(CONFIG_DIR, "tokenizer.json"))
        except Exception as e:
            self.tokenizer = None
            print("[tok] tokenizer unusable on this box (%s); ids in, ids out" % (e,), flush=True)
        t0 = time.time()
        self.net = xir.Network.CreateFromJsonFile(JSON_)
        self.net.lazyLoadParamsFromFile(RAW_)
        print("[net] json+raw %.1f s" % (time.time() - t0), flush=True)
        mem("net_loaded")

        self.gmap = load_graph_map(JSON_)
        print("[graph] inputs=%d dead=%s pair=%d" % (len(self.gmap["ins"]), self.gmap["dead"],
                                                     len(self.gmap["pair"])), flush=True)
        if len(self.gmap["pair"]) != self.gmap["n_past"]:
            raise SystemExit("KV pairing incomplete")

        self.specs = list(self.net.ops[0].outputs)
        self.n_in = len(self.specs)
        if self.n_in != 42:
            raise SystemExit("expected 42 declared inputs, got %d" % self.n_in)

        self.feed = []
        for s in self.specs:
            t = s.tensorType()
            dt = np.float32 if "fp(32)" in str(t.element_dtype) else np.float16
            self.feed.append(np.zeros(list(t.shape), dtype=dt))
        print("[buf] feed bytes=%.1f MiB" % (sum(a.nbytes for a in self.feed) / 1048576.0), flush=True)

        lg = list(self.net.ops[-1].inputs[0].tensorType().shape)
        self.logits = np.zeros(lg, dtype=np.float16)

        self.dev = openDevice("zg330", "192.168.125.143", True, False)
        mem("openDevice")
        t0 = time.time()
        self.sess = initSession("zg330", self.net, self.dev, 1)
        mem("initSession")
        t0 = time.time()
        self.sess.apply()
        print("[net] apply %.1f s" % (time.time() - t0), flush=True)
        mem("apply")

    def _tensors(self):
        # rows 0/1 take xir.Layout("*C") exactly like the two proven drivers on this board;
        # the cache rows take the network's own declared type
        out = []
        for i, (a, s) in enumerate(zip(self.feed, self.specs)):
            out.append(Tensor(a, xir.Layout("*C") if i < 2 else s.tensorType()))
        return out

    def step(self, token, past_len):
        f = self.feed
        f[0][0, 0] = np.float32(token)
        f[1][0, :] = np.asarray([1.0, past_len, 0.0], dtype=np.float32)
        t0 = time.time()
        outs = self.sess.forward(self._tensors())
        for t in outs:
            t.waitForReady(datetime.timedelta(milliseconds=50000))
        self.dev.reset(1)
        dt = time.time() - t0
        # logits
        ch = outs[0].chunk()
        if ch.byte_size != self.logits.nbytes:
            raise SystemExit("logits %d B != %d B" % (ch.byte_size, self.logits.nbytes))
        outs[0].read(self.logits, outs[0].offset(), ch.byte_size)
        # chain present -> matching past buffer; done after this step's numbers are out
        for oidx, pvid in self.gmap["pair"].items():
            buf = f[pvid]
            t = outs[oidx]
            ch = t.chunk()
            if ch.byte_size != buf.nbytes:
                raise SystemExit("out[%d] %d B != past v%d %d B"
                                 % (oidx, ch.byte_size, pvid, buf.nbytes))
            t.read(buf, t.offset(), ch.byte_size)
            if PADFIX:
                buf[:, :, 0, :] = buf[:, :, 1, :]
        return int(np.argmax(self.logits.reshape(-1))), dt


def main():
    print("=== ERNIE-4.5-0.3B E41 ZG330  padfix=%d max_new=%d ===" % (PADFIX, MAX_NEW), flush=True)
    m = E41()
    if TEXT:
        if m.tokenizer is None:
            raise SystemExit("E41_TEXT needs a working tokenizer; pass E41_PROMPT ids instead")
        prompt = m.tokenizer.encode(TEXT).ids
    else:
        prompt = [int(t) for t in PROMPT.split(",") if t.strip()]
    walk = ([prompt[0]] + prompt) if PADFIX else list(prompt)
    seq = list(prompt)
    feedt = list(walk)
    print("[gen] prompt tokens=%d %s" % (len(prompt), prompt), flush=True)
    print("[gen] walk=%d (padfix=%d)" % (len(walk), PADFIX), flush=True)

    record = []
    total = len(walk) + MAX_NEW
    for i in range(total):
        if i >= len(feedt):
            break
        past_len = i + POS_OFFSET - (1 if (PADFIX and i >= 1) else 0)
        if past_len >= MAX_SLOTS:
            print("cache full at step %d" % i, flush=True)
            break
        tok = feedt[i]
        nxt, dt = m.step(tok, past_len)
        if i >= len(walk) - 1:
            seq.append(nxt)
            feedt.append(nxt)
        lg = m.logits.reshape(-1).astype(np.float32)
        record.append({"step": i, "fed": tok, "past_len": past_len, "argmax": nxt,
                       "logits_min": float(lg.min()), "logits_max": float(lg.max()),
                       "seconds": round(dt, 2)})
        print("step %3d fed %-7d past %-4d -> %-7d (%.2fs, logits %.3f..%.3f)"
              % (i, tok, past_len, nxt, dt, lg.min(), lg.max()), flush=True)
        json.dump(record, open("/tmp/e41_steps.json", "w"), indent=1)
        if i >= len(walk) - 1 and nxt in STOP:
            print("stop token %d at step %d" % (nxt, i), flush=True)
            break
    gen = seq[len(prompt):]
    print("[gen] generated ids %s" % gen, flush=True)
    if m.tokenizer is not None:
        print("[gen] text %r" % m.tokenizer.decode(gen, skip_special_tokens=False), flush=True)
    out = os.environ.get("E41_OUT", "/tmp/e41_steps.json")
    json.dump(record, open(out, "w"), indent=1)
    mem("done")


if __name__ == "__main__":
    main()
