#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E41（ERNIE-4.5-0.3B 那颗 _ZG 件）的板侧分词：纯标准库，不碰 tokenizers／sentencepiece。

背景：板上 tokenizers 0.13.3 的 from_file 读不了这颗 tokenizer.json（台账 #220，成因未查明），
所以分词与解码此前只能在 PC 侧做（board/e41_chat_console.py 里 tok.encode／tok.decode）。
本件把那一步搬上板：数据是 scripts/mk_e41_tok.py 预打好的 board/e41_tok.pack（3,302,740 B），
运行时只读这个包，不把 11 MB JSON 的解析对象常驻在 daemon 里。

复刻的参照物 = 本机 tokenizers 0.23.3 读同一颗 tokenizer.json，几条关键口径都实测过：
  added_tokens 一律参与匹配，而且 add_special_tokens 真假对这份 tokenizer 没有区别
    （post_processor 是空 TemplateProcessing，不在首尾加任何东西）：begin_of_sentence 在
    两种调用下都出 100273；数字 0-9 是 non-special added token，所以 2026 逐位切成
    5,3,5,9 而不是被 merge 成整串。
  normalizer 只有一条 Replace 空格→U+2581，pre_tokenizer = null ⇒ 归一后的整段就是一个
    word，merge 可以跨过原来的空格（连续两个 U+2581 本身就是 merges[0]）。
  ignore_merges = False ⇒ 没有 whole-word 捷径；byte_fallback = True ⇒ 不在词表的字符按
    UTF-8 逐字节走 <0xHH>（词表里 256 个字节 token 全在，id 13..268）；换行与制表就是字节
    token（它们不是单字符词条）。
  decode = 逐 token 把 U+2581 换成空格、<0xHH> 还原成那个字节、末尾整串按 UTF-8 有损解码
    （等于参照物那三条 decoder：Replace／ByteFallback／Fuse 的合成效果）。

校验：scripts/tok_parity.py 拿本机 tokenizers 当参照物逐条比 ids 与解码串，实测计数写在
该文件末尾。**板端零执行**：板子还没回来，包没上板，板上那颗 Python 3.8 也没跑过本件。

用法：
    python3 e41_tok.py "文本"                打 ids（读同目录的 e41_tok.pack）
    python3 e41_tok.py --ids 100273,104,2    打解码串
    python3 e41_tok.py --selftest            载入包＋几条自检与耗时
程序里调：
    import e41_tok
    tk = e41_tok.load()                    # 或 load("/mnt/data/ernie/e41_tok.pack")
    ids = tk.encode(prompt_text)
    text = tk.decode(gen_ids)
"""
import hashlib
import os
import struct
import sys
import zlib

MAGIC = b"E41TOK01"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PACK = os.path.join(HERE, "e41_tok.pack")
MARK = "▁"
SPACE = " "
ID_BASE = 1 << 17


class Tok(object):

    def __init__(self, vocab, id2tok, special, char2id, pair, byte_ids, byte_of, added, unk,
                 rank_bits, n_ids, src_md5, pack):
        self.vocab = vocab
        self.id2tok = id2tok
        self.special = special
        self.char2id = char2id
        self.pair = pair
        self.byte_ids = byte_ids
        self.byte_of = byte_of
        self.added = added
        self.unk = unk
        self.rb = rank_bits
        self.rmask = (1 << rank_bits) - 1
        self.n_ids = n_ids
        self.base = ID_BASE
        self.src_md5 = src_md5
        self.pack = pack

    def _bpe(self, seq):
        """贪心 BPE：每轮取 rank 最小（同 rank 取最左）的那一对，整串一次性并掉所有出现。"""
        pair = self.pair
        rmask = self.rmask
        rb = self.rb
        base = self.base
        n = len(seq)
        while n > 1:
            best_rank = -1
            best_i = -1
            best_new = 0
            for i in range(n - 1):
                v = pair.get(seq[i] * base + seq[i + 1])
                if v is not None:
                    r = v & rmask
                    if best_rank < 0 or r < best_rank:
                        best_rank = r
                        best_i = i
                        best_new = v >> rb
            if best_i < 0:
                break
            left = seq[best_i]
            right = seq[best_i + 1]
            out = []
            i = 0
            while i < n:
                if i + 1 < n and seq[i] == left and seq[i + 1] == right:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
            n = len(seq)
        return seq

    def _chunk(self, text):
        """一段不含 added token 的原文：先归一（空格→U+2581），再按字符查词表，
        查不到的字符单独走字节回退（它进不了任何 merge，所以先切断是等价的）。"""
        out = []
        run = []
        char2id = self.char2id
        byte_ids = self.byte_ids
        for ch in text.replace(SPACE, MARK):
            cid = char2id.get(ch)
            if cid is None:
                if run:
                    out.extend(self._bpe(run))
                    del run[:]
                for b in ch.encode("utf-8"):
                    out.append(byte_ids[b])
            else:
                run.append(cid)
        if run:
            out.extend(self._bpe(run))
        return out

    def encode(self, text):
        out = []
        adict = self.added
        n = len(text)
        i = 0
        start = 0
        while i < n:
            hit = None
            cands = adict.get(text[i])
            if cands is not None:
                for tok, tid, size in cands:
                    if text.startswith(tok, i):
                        hit = (tid, size)
                        break
            if hit is None:
                i += 1
                continue
            if i > start:
                out.extend(self._chunk(text[start:i]))
            out.append(hit[0])
            i += hit[1]
            start = i
        if n > start:
            out.extend(self._chunk(text[start:]))
        return out

    def decode(self, ids, skip_special_tokens=True):
        out = []
        buf = bytearray()
        id2tok = self.id2tok
        special = self.special
        byte_of = self.byte_of
        n_ids = self.n_ids

        def flush():
            if not buf:
                return
            try:
                out.append(buf.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("\ufffd" * len(buf))
            del buf[:]

        for tid in ids:
            if tid >= n_ids:
                raise ValueError("id %d 不在这份词表的 id 空间 %d 里" % (tid, n_ids))
            tok = id2tok[tid]
            if tok is None:
                continue
            if skip_special_tokens and special[tid]:
                continue
            b = byte_of.get(tok)
            if b is None:
                flush()
                out.append(tok.replace(MARK, SPACE))
            else:
                buf.append(b)
        flush()
        return "".join(out)

    def info(self):
        return {"pack": self.pack,
                "pack_md5": hashlib.md5(open(self.pack, "rb").read()).hexdigest(),
                "src_tokenizer_json_md5": self.src_md5,
                "vocab": len(self.vocab), "id_space": self.n_ids,
                "added": sum(len(v) for v in self.added.values()),
                "merges": len(self.pair), "unk": self.unk,
                "space_mark": "U+2581"}


def _reader(raw):
    off = [0]

    def u32s(k=1):
        o = off[0]
        off[0] = o + 4 * k
        return struct.unpack_from("<%dI" % k, raw, o) if k > 1 else struct.unpack_from("<I", raw, o)[0]

    def s():
        o = off[0]
        n = struct.unpack_from("<I", raw, o)[0]
        o += 4
        off[0] = o + n
        return raw[o:off[0]].decode("utf-8")
    return u32s, s


def load(path=None):
    path = path or os.environ.get("E41_TOK_PACK") or DEFAULT_PACK
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        raise ValueError("包魔数不对：%r（不是 %r）" % (raw[:8], MAGIC))
    crc, size = struct.unpack_from("<II", raw, 8)
    body = raw[16:16 + size]
    if len(body) != size:
        raise ValueError("包短了：%d < %d" % (len(body), size))
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ValueError("包 crc 不过：内容坏了或没读全")
    u32s, s = _reader(body)
    n_ids = u32s()
    if n_ids > ID_BASE:
        raise ValueError("id 空间 %d 大于 pair 键基数 %d，这张表会撞键" % (n_ids, ID_BASE))
    unk = u32s()
    n_special = u32s()
    special = bytearray(n_ids)
    for tid in u32s(n_special):
        special[tid] = 1
    n_vocab = u32s()
    vocab = []
    id2tok = [None] * n_ids
    char2id = {}
    for i in range(n_vocab):
        tok = s()
        vocab.append(tok)
        id2tok[i] = tok
        if len(tok) == 1:
            char2id[tok] = i
    n_added = u32s()
    added = {}
    for _ in range(n_added):
        tid = u32s()
        tok = s()
        if id2tok[tid] is None:
            id2tok[tid] = tok
        first = tok[0]
        lst = added.setdefault(first, [])
        lst.append((tok, tid, len(tok)))
    for lst in added.values():
        lst.sort(key=lambda x: -x[2])
    n_bytes = u32s()
    byte_ids = list(u32s(n_bytes))
    byte_of = {}
    for b in range(n_bytes):
        byte_of["<0x%02X>" % b] = b
    n_merges = u32s()
    rank_bits = u32s()
    base = ID_BASE
    pair = {}
    for r in range(n_merges):
        left, right, new = u32s(3)
        pair[left * base + right] = (new << rank_bits) | r
    src_md5 = s()
    return Tok(vocab, id2tok, special, char2id, pair, byte_ids, byte_of, added, unk,
               rank_bits, n_ids, src_md5, path)


def main(argv):
    args = argv[1:]
    path = None
    if args and args[0].endswith(".pack"):
        path = args.pop(0)
    tk = load(path)
    if not args or args[0] == "--selftest":
        i = tk.info()
        print("pack      %s" % i["pack"])
        print("md5       %s（src tokenizer.json %s）" % (i["pack_md5"], i["src_tokenizer_json_md5"]))
        print("词表      %d | id 空间 %d | merges %d | added %d | unk %d"
              % (i["vocab"], i["id_space"], i["merges"], i["added"], i["unk"]))
        probe = "User: 你好\nAssistant: "
        ids = tk.encode(probe)
        assert tk.decode(ids) == probe, "换行往返对不上"
        assert tk.encode("2026")[0] == 5, "数字没按 added token 切"
        assert tk.encode("<|begin_of_sentence|>") == [100273], "BOS 没出单 token"
        want = [tk.char2id["a"]] + [tk.byte_ids[b] for b in "😀".encode("utf-8")] + [tk.char2id["b"]]
        assert tk.encode("a😀b") == want, "字节回退不是逐 UTF-8 字节"
        assert tk.decode(tk.encode("a😀b")) == "a😀b", "emoji 往返对不上"
        import time
        t0 = time.time()
        big = (probe * 40)
        n = len(tk.encode(big))
        dt = time.time() - t0
        print("自检      往返过 | 数字过 | BOS 过 | %d 字 %d token 用 %.2f s（本机）"
              % (len(big), n, dt))
        return 0
    if args[0] == "--ids":
        ids = [int(x) for x in args[1].split(",") if x != ""]
        print(tk.decode(ids))
        return 0
    text = args[0]
    ids = tk.encode(text)
    print(len(ids))
    print(",".join(str(x) for x in ids))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
