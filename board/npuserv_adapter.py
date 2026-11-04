#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""npuserv adapter: E41-decoder_ZG (ERNIE-4.5-0.3B) on the ZG330.

Model-specific steps are NOT reimplemented here -- they stay in `e41_chatd.py`, the file
that is validated on the board (台账 §廿六／§册一: padfix, POS_OFFSET, the 511-slot policy
gate, narrow KV readback, MAX_NEW_CAP=192, stop {2,100272}).  This adapter only translates
the npuserv request shape into that engine's shape.

Verification status (2026-11-04): the e41_chatd engine is A (ran on the board, many
turns).  This wrapper has NOT been run on the board -- the board is not back.  Nothing in
this file changes what gets fed to the NPU.

Board layout: push this file next to e41_chatd.py, same directory, so `import e41_chatd`
resolves; the registry path is relative to lang/llm and the server runs from a directory
that mirrors lang/llm (see npuserv/README.md).

Request (POST /generate): {"ids":[...], "max_new":96}
                          {"ids":[...], "max_new":96, "max_slots":512}   # see below
Text is refused: this 件 takes ids in / ids out, encode+decode stay with the caller
(why: 台账 #220 -- board tokenizers 0.13.3 fails to load its tokenizer.json; the cause is
unattributed, board-side tokenization is not ruled out, it just has not been built).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.environ.get("NPUSERV_HOME", "")):
    if path and path not in sys.path:
        sys.path.insert(0, path)

import adapter_lib                                             # noqa: E402


def _apply_env(params):
    """npuserv params -> the env vars e41_chatd reads at import time.

    Anything unset falls through to e41_chatd's own default, so a bare POST /load gives
    the 现役 product configuration, not a new one invented here.
    """
    mapping = {
        "json": "E41_JSON", "raw": "E41_RAW",
        "pos_offset": "E41_POS_OFFSET", "max_new_cap": "E41_MAX_NEW_CAP",
    }
    for key, env in mapping.items():
        if params.get(key) is not None:
            os.environ[env] = str(params[key])
    if params.get("padfix") is not None:
        os.environ["E41_PADFIX"] = "1" if params["padfix"] else "0"
    if params.get("stop_ids"):
        os.environ["E41_STOP"] = ",".join(str(int(t)) for t in params["stop_ids"])
    # The daemon talks AF_UNIX; npuserv talks loopback TCP through adapter_lib.
    # Nothing here binds a socket, but keep the paths out of the way of a 现役 daemon.
    os.environ.setdefault("E41_SOCK", "/tmp/npuserv_e41.sock")
    os.environ.setdefault("E41_LOCK", "/tmp/npuserv_e41.lock")


class ErnieEngine(object):
    name = "ernie-4.5-0.3b"

    def __init__(self, M, m, params):
        self.M = M
        self.m = m
        self.params = params
        self.apply_s = m.apply_s
        self.info = {
            "max_slots_policy": M.MAX_SLOTS,
            "kv_heads": m.kv_heads, "kv_depth": m.kv_depth,
            "kv_slot_bytes": m.kv_slot_b,
            "declared_inputs": len(m.specs), "past_inputs": m.gmap["n_past"],
            "padfix": M.PADFIX, "pos_offset": M.POS_OFFSET,
            "stop_ids": sorted(M.STOP), "max_new_cap": M.MAX_NEW_CAP,
            "read": "narrow", "json": M.JSON_, "raw": M.RAW_,
        }

    @property
    def n_steps(self):
        return self.m.n_steps

    def generate(self, req, emit):
        if req.get("text") and not req.get("ids"):
            raise ValueError("this model takes ids; encode stays with the caller "
                             "(capabilities.text_in=false, see GET /models)")
        ids = [int(t) for t in req.get("ids") or []]
        if not ids:
            raise ValueError("ids required")
        max_new = int(req.get("max_new", self.params.get("max_new", 96)))
        box = {}

        def capture(ev):
            if ev.get("end"):
                box.update(ev)
            else:
                emit(ev)

        self.m.generate(ids, max_new, capture)
        out = dict(box)
        out.setdefault("ids", [])
        out["model"] = self.name
        return out

    def close(self):
        # The session dies with the process; there is no documented release call, and
        # 台账 §册三 says CMA comes back only when the process exits.  So: do nothing.
        pass


def load(params):
    _apply_env(params)
    alt = params.get("e41_dir")
    if alt:
        # This adapter's own directory goes on sys.path first, because that is the board
        # layout (it lives next to e41_chatd.py).  A fixture -- or any layout where the
        # engine lives elsewhere -- has to name its directory explicitly.
        sys.path.insert(0, alt)
    import e41_chatd as M
    override = params.get("max_slots")
    if override is not None:
        # Policy gate only. e41_chatd hard-codes 511 (one slot short of the compiled
        # NS=512 window); 台账 §册三 left "can this 件 swallow a full 512-slot prompt"
        # unmeasured -- task #24. Raising it here is that measurement's entry point;
        # the hard guard in E41.step (past_len >= kv_depth) still stops at 511.
        M.MAX_SLOTS = int(override)
    adapter_lib.log("importing e41_chatd and applying the net")
    m = M.E41()
    return ErnieEngine(M, m, params)


if __name__ == "__main__":
    sys.exit(adapter_lib.run(load))
