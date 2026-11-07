#!/usr/bin/env python3
"""ERNIE-4.5-0.3B (E41 ZG) chat daemon for the ZG330 board.

Loads the net once (json+raw+apply, ~50-95 s), then serves JSON lines over a
unix socket so each chat turn costs only forward passes.  Exits after
E41_IDLE seconds with no client attached (default 300) so the board is not
held forever.

  python3 e41_chatd.py serve      # daemon; `connect` spawns it when missing
  python3 e41_chatd.py connect    # relay stdin/stdout to the socket

Wire format (one JSON object per line, both directions):
  ->  {"ids":[...], "max_new":96}
  <-  {"ready":true,"apply_s":46.4}
  <-  {"pre":3,"n":21}                     prompt token i of n is done
  <-  {"tok":8231,"i":0,"sec":0.42}        generated token i
  <-  {"end":true,"ids":[...],"reason":"eos|limit|full","sec":13.1}
  ->  {"bye":true}

Every request replays the whole prompt with zeroed KVs: that is the exact path
validated on 2026-11-01, and it keeps turns independent.
"""
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time

# The runtime (icraft/numpy) prints to stdout while it binds ops, so it is imported
# only by `serve`.  A relay that imported it would corrupt the JSON stream.
np = xir = xrt = Tensor = openDevice = initSession = datetime = None


def load_runtime():
    global np, xir, xrt, Tensor, openDevice, initSession, datetime
    import datetime as _dt
    import numpy as _np
    import icraft as _ic
    from icraft import xir as _xir
    from icraft import xrt as _xrt

    sys.path.append("/root/qwen2/modelzoo_utils")
    from pyrtutils.icraft_utils import openDevice as _od, initSession as _is

    np, xir, xrt, datetime = _np, _xir, _xrt, _dt
    Tensor, openDevice, initSession = _xrt.Tensor, _od, _is
    return _ic


JSON_ = os.environ.get("E41_JSON",
                     "/root/llm/models/ERNIE-4.5-0.3B-PT4icraft430tai/jr_path/decode/E41-decoder_ZG.json")
RAW_ = os.environ.get("E41_RAW",
                     "/root/llm/models/ERNIE-4.5-0.3B-PT4icraft430tai/jr_path/decode/E41-decoder_ZG.raw")
SOCK = os.environ.get("E41_SOCK", "/tmp/e41_chat.sock")
LOCK = os.environ.get("E41_LOCK", "/tmp/e41_chat.lock")
LOG = os.environ.get("E41_LOG", "/tmp/e41_chatd.log")
PADFIX = os.environ.get("E41_PADFIX", "1") == "1"
STOP = {int(t) for t in os.environ.get("E41_STOP", "2,100272").split(",") if t.strip()}
POS_OFFSET = int(os.environ.get("E41_POS_OFFSET", "1"))
MAX_SLOTS = 511
IDLE = int(os.environ.get("E41_IDLE", "300"))
MAX_NEW_CAP = int(os.environ.get("E41_MAX_NEW_CAP", "192"))


def log(msg):
    sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


def load_graph_map(jsonpath):
    """present v_id -> the past v_id it was sliced from; dead = declared, unused."""
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
    def __init__(self, device=None):
        load_runtime()
        t0 = time.time()
        self.net = xir.Network.CreateFromJsonFile(JSON_)
        self.net.lazyLoadParamsFromFile(RAW_)
        log("json+raw %.1f s" % (time.time() - t0))
        self.gmap = load_graph_map(JSON_)
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
        self.logits = np.zeros(list(self.net.ops[-1].inputs[0].tensorType().shape),
                               dtype=np.float16)
        # Narrow readback wants each KV feed seen as [head, slot, fragment bytes].
        g = list(self.specs[2].tensorType().shape)
        self.kv_heads, self.kv_depth, frag = int(g[1]), int(g[2]), int(g[3])
        self.kv_slot_b = frag * self.feed[2].itemsize
        self.kv_view = {}
        for pvid in set(self.gmap["pair"].values()):
            if list(self.specs[pvid].tensorType().shape)[1:] != g[1:] \
                    or self.feed[pvid].itemsize != self.feed[2].itemsize:
                raise SystemExit("KV v%d shape %s/%d B != %s"
                                 % (pvid, list(self.specs[pvid].tensorType().shape),
                                    self.feed[pvid].itemsize, g))
            self.kv_view[pvid] = self.feed[pvid].view(np.uint8).reshape(
                self.kv_heads, self.kv_depth, self.kv_slot_b)
        # multi-resident: device may be injected by npuserv (openDevice once per
        # process; a second handle corrupts other sessions' DMA - 2026-11-08)
        self.dev = device if device is not None else openDevice(
            "zg330", "192.168.125.143", True, False)
        self.sess = initSession("zg330", self.net, self.dev, 1)
        t0 = time.time()
        self.sess.apply()
        self.apply_s = time.time() - t0
        log("apply %.1f s" % self.apply_s)
        self.n_steps = 0

    def _tensors(self):
        return [Tensor(a, xir.Layout("*C") if i < 2 else s.tensorType())
                for i, (a, s) in enumerate(zip(self.feed, self.specs))]

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
        ch = outs[0].chunk()
        if ch.byte_size != self.logits.nbytes:
            raise SystemExit("logits %d B != %d B" % (ch.byte_size, self.logits.nbytes))
        outs[0].read(self.logits, outs[0].offset(), ch.byte_size)
        if past_len >= self.kv_depth:
            raise SystemExit("past_len %d >= cache depth %d" % (past_len, self.kv_depth))
        for oidx, pvid in self.gmap["pair"].items():
            t = outs[oidx]
            base = t.offset()
            dst = self.kv_view[pvid]
            for h in range(self.kv_heads):
                t.read(dst[h, past_len], base + (h * self.kv_depth + past_len) * self.kv_slot_b,
                       self.kv_slot_b)
            if PADFIX:
                buf = f[pvid]
                buf[:, :, 0, :] = buf[:, :, 1, :]
        self.n_steps += 1
        return int(np.argmax(self.logits.reshape(-1))), dt

    def reset(self):
        for i in range(2, len(self.feed)):
            self.feed[i].fill(0)

    def generate(self, ids, max_new, emit):
        self.reset()
        walk = ([ids[0]] + list(ids)) if PADFIX else list(ids)
        seq = list(ids)
        feedt = list(walk)
        max_new = max(1, min(max_new, MAX_NEW_CAP, MAX_SLOTS - len(walk) - 1))
        t0 = time.time()
        reason = "limit"
        gen = []
        total = len(walk) + max_new
        last_pre = 0
        for i in range(total):
            past_len = i + POS_OFFSET - (1 if (PADFIX and i >= 1) else 0)
            if past_len >= MAX_SLOTS:
                reason = "full"
                break
            tok = feedt[i]
            nxt, dt = self.step(tok, past_len)
            if i < len(walk) - 1:
                if i - last_pre >= 4:
                    last_pre = i
                    emit({"pre": i + 1, "n": len(walk)})
                continue
            seq.append(nxt)
            feedt.append(nxt)
            gen.append(nxt)
            emit({"tok": nxt, "i": len(gen) - 1, "sec": round(dt, 2)})
            if nxt in STOP:
                reason = "eos"
                break
        emit({"end": True, "ids": gen, "reason": reason,
              "sec": round(time.time() - t0, 1), "prompt": len(ids)})


def send(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def serve():
    m = E41()
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ls.bind(SOCK)
    os.chmod(SOCK, 0o600)
    ls.listen(1)
    log("listening %s" % SOCK)
    idle_since = time.time()
    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        while not stop["flag"]:
            r, _, _ = select.select([ls], [], [], 5.0)
            if not r:
                if time.time() - idle_since > IDLE:
                    log("idle %d s, exiting" % IDLE)
                    break
                continue
            conn, _ = ls.accept()
            idle_since = time.time()
            buf = b""
            try:
                send(conn, {"ready": True, "apply_s": round(m.apply_s, 1),
                            "padfix": PADFIX})
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        req = json.loads(line.decode("utf-8"))
                        if req.get("bye"):
                            raise EOFError
                        ids = [int(t) for t in req["ids"]]
                        if not ids:
                            send(conn, {"err": "empty ids"})
                            continue
                        m.generate(ids, int(req.get("max_new", 96)),
                                   lambda ev: send(conn, ev))
            except EOFError:
                pass
            except Exception as e:
                log("client error: %r" % (e,))
                try:
                    send(conn, {"err": repr(e)})
                except Exception:
                    pass
            finally:
                conn.close()
                idle_since = time.time()
                log("client gone, steps total=%d" % m.n_steps)
    finally:
        try:
            ls.close()
            os.unlink(SOCK)
        except Exception:
            pass
        try:
            os.unlink(LOCK)
        except Exception:
            pass
        log("daemon exit")


def sock_alive():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(SOCK)
        return s
    except Exception:
        s.close()
        return None


def spawn_daemon():
    with open(LOG, "ab") as f:
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "serve"],
                             stdin=subprocess.DEVNULL, stdout=f, stderr=f,
                             start_new_session=True)
    open(LOCK, "w").write(str(p.pid))
    return p.pid


def connect():
    s = sock_alive()
    if s is None:
        pid = spawn_daemon()
        sys.stdout.write(json.dumps({"status": "starting", "pid": pid}) + "\n")
        sys.stdout.flush()
        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(1.0)
            s = sock_alive()
            if s is not None:
                break
        if s is None:
            sys.stderr.write("daemon did not come up; see %s\n" % LOG)
            return 2
    s.settimeout(None)
    import threading

    def pump_in():
        try:
            while True:
                data = os.read(0, 65536)
                if not data:
                    break
                s.sendall(data)
        except Exception:
            pass
        finally:
            try:
                s.shutdown(socket.SHUT_WR)
            except Exception:
                pass

    t = threading.Thread(target=pump_in, daemon=True)
    t.start()
    try:
        while True:
            data = s.recv(65536)
            if not data:
                break
            os.write(1, data)
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        serve()
    elif cmd == "connect":
        sys.exit(connect())
    else:
        raise SystemExit("usage: e41_chatd.py [serve|connect]")
