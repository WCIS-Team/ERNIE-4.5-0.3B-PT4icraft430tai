# -*- coding: utf-8 -*-
"""ERNIE-4.5-0.3B 板端对话入口（ZG330 / 192.168.1.113）。

分词与解码在本机做（板上 tokenizers 0.13.3 读不了这颗 tokenizer.json），
板端服务 e41_chatd.py 常驻，模型只装一次。首次连接若服务没起，会自动拉起（约 50 s）。

用法：双击本文件，或 `python "ERNIE对话.py"`。
命令：/clear 清空上下文   /max N 改生成长度   /stats 上一轮速度   /exit 退出
"""
import json
import os
import subprocess
import sys
import time

BOARD = "root@192.168.1.113"
REMOTE = "/mnt/data/ernie/e41_chatd.py"
TOK_JSON = r"S:\work\WCIS\repos\ERNIE-4.5-0.3B-PT4icraft430tai\export\ernie_fp16_cuda\tokenizer.json"
BOS = "<|begin_of_sentence|>"
EOS = "<|end_of_sentence|>"
DEFAULT_MAX_NEW = 96

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleTitleW("ERNIE 对话（ZG330 板端）")
except Exception:
    pass


def say(s=""):
    print(s, flush=True)


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOK_JSON)


def render(history):
    """same template as export/ernie_fp16_cuda/chat_template.jinja"""
    s = BOS
    for m in history:
        if m["role"] == "user":
            s += "User: " + m["content"] + "\n"
        elif m["role"] == "assistant":
            s += "Assistant: " + m["content"] + EOS
        else:
            s += m["content"] + "\n"
    return s + "Assistant: "


def read_lock():
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "deb-h61",
                              "cat /home/user/Desktop/113.txt"],
                             capture_output=True, text=True, timeout=20,
                             stdin=subprocess.DEVNULL)
        return (out.stdout or "").strip()
    except Exception:
        return None


def write_lock(value):
    try:
        subprocess.run(["ssh", "-o", "ConnectTimeout=8", "deb-h61",
                        "printf '%s' > /home/user/Desktop/113.txt" % value],
                       capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
    except Exception:
        pass


class Board:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["ssh", "-T", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
             BOARD, "python3", "-u", REMOTE, "connect"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.apply_s = None

    def read_until_ready(self):
        t0 = time.time()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = (self.proc.stderr.read() or "").strip()
                raise RuntimeError("板端服务没起来：%s" % (err[-400:] or "无输出"))
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("status") == "starting":
                say("板上服务启动中（载图约 50 s ＋ apply 约 45 s，请稍候）…")
            elif ev.get("ready"):
                self.apply_s = ev.get("apply_s")
                return ev
            elif "err" in ev:
                raise RuntimeError(ev["err"])

    def ask(self, ids, max_new, on_pre, on_tok):
        req = json.dumps({"ids": ids, "max_new": max_new}) + "\n"
        self.proc.stdin.write(req)
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("连接断了（板端服务退出或被占用）")
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if "pre" in ev:
                on_pre(ev["pre"], ev["n"])
            elif "tok" in ev:
                on_tok(ev["tok"], ev)
            elif "end" in ev:
                return ev
            elif "err" in ev:
                raise RuntimeError(ev["err"])

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"bye": True}) + "\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main():
    say("=== ERNIE-4.5-0.3B · ZG330 板端对话 ===")
    say("命令：/clear 清空上下文  /max N 生成长度  /stats 速度  /exit 退出")
    tok = load_tokenizer()

    prev_lock = read_lock()
    we_locked = False
    if prev_lock is None:
        say("注意：查不到 113.txt（deb-h61 不通），板子占用状态未知。")
    elif prev_lock == "test==on":
        say("注意：113.txt = test==on，板上已有别的会话在用，继续可能冲突。")
        ans = input("仍要连接？[y/N] ").strip().lower()
        if ans != "y":
            return
    else:
        write_lock("test==on")
        we_locked = read_lock() == "test==on"

    board = Board()
    try:
        ev = board.read_until_ready()
        MAX_SLOTS = ev["max_slots"] - 11          # NS=512 -> 500，与旧常数一致
        say("已连上板端服务（apply %s s，padfix=%s，窗口 %s 槽）"
            % (ev.get("apply_s"), ev.get("padfix"), ev.get("max_slots")))
        say("")
        history = []
        max_new = DEFAULT_MAX_NEW
        last = {"sec": 0, "n": 0}
        while True:
            try:
                q = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                say()
                break
            if not q:
                continue
            if q in ("/exit", "/quit", "exit", "quit"):
                break
            if q == "/clear":
                history = []
                say("(上下文已清空)")
                continue
            if q == "/stats":
                if last["n"]:
                    say("上一轮 %d token / %s s = %.2f s/token" %
                        (last["n"], last["sec"], last["sec"] / last["n"]))
                else:
                    say("(还没有生成过)")
                continue
            if q.startswith("/max"):
                try:
                    max_new = max(1, min(400, int(q.split()[1])))
                    say("(生成长度上限 %d)" % max_new)
                except Exception:
                    say("用法：/max 128")
                continue

            history.append({"role": "user", "content": q})
            while True:
                ids = tok.encode(render(history), add_special_tokens=False).ids
                if len(ids) + 16 <= MAX_SLOTS:
                    break
                if len(history) <= 1:
                    ids = ids[-(MAX_SLOTS - 32):]
                    break
                history = history[1:]
                say("(上下文超长，丢掉最早一轮)")

            say("ERNIE> ", )
            gen = []
            shown = ""

            def on_pre(k, n):
                sys.stdout.write("\r[读提示 %d/%d] " % (k, n))
                sys.stdout.flush()

            def on_tok(tid, ev):
                nonlocal shown
                gen.append(tid)
                text = tok.decode(gen, skip_special_tokens=True)
                if len(text) > len(shown):
                    sys.stdout.write(text[len(shown):])
                    sys.stdout.flush()
                    shown = text

            t0 = time.time()
            ev = board.ask(ids, max_new, on_pre, on_tok)
            if shown == "":
                sys.stdout.write("(空)")
            sys.stdout.write("\n")
            sys.stdout.flush()
            answer = tok.decode(gen, skip_special_tokens=True).strip()
            history.append({"role": "assistant", "content": answer})
            last = {"sec": ev.get("sec", time.time() - t0), "n": len(gen)}
            say("    [%d token · %s s · %s]" % (len(gen), last["sec"], ev.get("reason")))
            say("")
    finally:
        try:
            board.close()
        except Exception:
            pass
        if we_locked:
            write_lock(prev_lock)
            say("(113.txt 已恢复 %s)" % (prev_lock or "test==off"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        say("出错：%s" % e)
    finally:
        try:
            input("\n回车退出…")
        except Exception:
            pass
