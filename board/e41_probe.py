"""ERNIE-4.5-0.3B E41 ZG pre-flight on the board: staged memory/segment measurement before
spending an apply().  E41STEP: 0=load net only, 1=+openDevice, 2=+initSession, 3=+apply.

Nothing here deletes or rewrites model files.  Same shape as qwen3/board/q3_probe.py, which ran
green on 10-14 for Q3V1; the only new thing this prints is the E41 input row (42 declared values,
4 of which have no consumer).
"""
import os, sys, time, resource

sys.path.append("/root/qwen2/modelzoo_utils")
import icraft
from icraft import xir, xrt, zg330backend
from pyrtutils.icraft_utils import openDevice, initSession

STEP = int(os.environ.get("E41STEP", "2"))
JSON_ = "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.json"
RAW_ = "/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.raw"


def mem(tag):
    st = dict()
    for line in open("/proc/self/status"):
        k = line.split(":")[0]
        if k in ("VmSize", "VmRSS", "VmHWM", "VmSwap"):
            st[k] = line.split()[1] + " " + line.split()[2]
    cm = {}
    for line in open("/proc/meminfo"):
        k = line.split(":")[0]
        if k in ("MemAvailable", "MemFree", "CmaFree", "Cached", "SwapFree"):
            cm[k] = line.split()[1]
    print("[MEM] %-12s %s | %s" % (
        tag, " ".join("%s=%s" % (k, v) for k, v in st.items()),
        " ".join("%s=%s" % (k, v) for k, v in cm.items())), flush=True)


def kb(n):
    return "%.1f MiB" % (n / 1048576.0)


def describe(o, i):
    try:
        t = o.tensorType()
        shape = list(t.shape)
        dt = str(t.element_dtype)
    except Exception as e:
        shape, dt = "?%r" % (e,), "?"
    vid = getattr(o, "v_id", None)
    return "  in[%02d] v_id=%s shape=%s dtype=%s" % (i, vid, shape, dt)


print("=== e41_probe STEP=%d pid=%d ===" % (STEP, os.getpid()), flush=True)
mem("start")

net = xir.Network.CreateFromJsonFile(JSON_)
mem("create_json")
net.lazyLoadParamsFromFile(RAW_)
mem("lazy_params")
pb = getattr(net, "params_bytes", None)
print("params_bytes=%r" % (pb() if callable(pb) else pb), flush=True)

ins = net.ops[0].outputs
outs = net.ops[-1].inputs if hasattr(net.ops[-1], "inputs") else None
print("input op outputs=%d  last op inputs=%s" % (len(ins), len(outs) if outs is not None else "?"), flush=True)
for i, o in enumerate(ins):
    print(describe(o, i), flush=True)
for i, o in enumerate(outs or []):
    try:
        t = o.tensorType()
        print("  out[%02d] shape=%s dtype=%s" % (i, list(t.shape), t.element_dtype), flush=True)
    except Exception as e:
        print("  out[%02d] ?%r" % (i, e), flush=True)

if STEP < 1:
    print("stop before openDevice (E41STEP=%d)" % STEP, flush=True)
    sys.exit(0)
dev = openDevice("zg330", "192.168.125.143", True, False)
mem("openDevice")


def resolve(v):
    if not hasattr(v, "__int__"):
        for a in ("toString ", "toString"):
            if hasattr(v, a):
                try:
                    return "%s()" % a + str(getattr(v, a)())
                except Exception as e:
                    return "toString err %r" % (e,)
        return "noref-attrs=%s" % [x for x in dir(v) if not x.startswith("_")]
    try:
        return int(v)
    except Exception as e:
        return "int() -> %r" % (e,)


for nm in ("plddr", "udma", "ocm"):
    try:
        reg = dev.getMemRegion(nm)
        info = reg.memManager().getMemRegionInfo()
        print("[REGION] %s info=%s" % (nm, {k: resolve(x) for k, x in info.items()}), flush=True)
        mm = reg.memManager()
        chunks = mm.getAllMemChunk()
        print("[REGION] %s chunks=%d" % (nm, len(chunks)), flush=True)
        for ch in chunks[:4]:
            print("   chunk %s" % ({k: getattr(ch, k) for k in ("begin", "size", "byte_size") if hasattr(ch, k)}), flush=True)
    except Exception as e:
        print("[REGION] %s -> %r" % (nm, e), flush=True)

if STEP < 2:
    print("stop before initSession (E41STEP=%d)" % STEP, flush=True)
    sys.exit(0)
sess = initSession("zg330", net, dev, 1)
mem("session_create")
zg = zg330backend.ZG330Backend(sess.backends[0])

SEG = {0: "WEIGHT", 1: "INSTR", 2: "FTMP", 3: "INPUT", 4: "OUTPUT"}


def num(v):
    r = resolve(v)
    return r if isinstance(r, int) else None


def dump(segmap, label):
    tot = 0
    hi = 0
    for st, seg in segmap.items():
        b = num(seg.byte_size)
        a = num(getattr(seg, "phy_addr", None))
        if a is None:
            a = num(getattr(seg, "logic_addr", None))
        if a is None or b is None:
            print("  seg %s unreadable: byte_size=%s addr=%s attrs=%s" % (
                st, resolve(seg.byte_size), resolve(getattr(seg, "phy_addr", None)),
                [x for x in dir(seg) if not x.startswith("_")]), flush=True)
            continue
        tot += b
        hi = max(hi, a + b)
        print("  [%s] type=%s addr=0x%x byte_size=%s (%s)" % (label, SEG.get(int(st), st), a, b, kb(b)), flush=True)
    print("[%s] TOTAL=%s  highest_addr=0x%x (%s)" % (label, kb(tot), hi, kb(hi)), flush=True)
    return tot


dump(zg.logic_segment_map, "logic")
if STEP < 3:
    print("skip apply (E41STEP=%d)" % STEP, flush=True)
    sys.exit(0)
t0 = time.time()
sess.apply()
print("apply %.1f s" % (time.time() - t0), flush=True)
mem("apply")
dump(zg.phy_segment_map, "phy")
try:
    print("precheck=%s" % zg.precheck(), flush=True)
except Exception as e:
    print("precheck -> %r" % (e,), flush=True)
print("=== done ===", flush=True)
