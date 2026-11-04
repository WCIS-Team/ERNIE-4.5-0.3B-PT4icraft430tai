#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 tokenizer.json 打成板侧分词包 board/e41_tok.pack。

为什么要预打包：板上 tokenizers 0.13.3 读不了这颗 tokenizer.json（台账 #220），而
board/e41_tok.py 只用标准库、不碰 tokenizers／sentencepiece；tokenizer.json 本身 11 MB，
本机 CPython 3.13 用 stdlib json 直接 load 实测 0.33 s／RSS 19→68 MB，这开销不能跟着
常驻 daemon 走（板子 993 MB，LLM 那侧 VmHWM 已在 500 MB 量级）⇒ 预打成约 2 MB 的二进制包。

包里的每个数都来自 tokenizer.json 实测，不是推测（本机 tokenizers 0.23.3 读数）：
  vocab 100,295 条，id 0..100294 连续；merges 182,926 条，按 rank 顺序，
  左右串与结果串都在 vocab 内、pair 无重复（所以按 (左id,右id) 建表是安全的）；
  added_tokens 1,029 条：只有 20 条在 vocab 里（<unk>／<s>／</s>／数字 0-9／
  <|end_of_sentence|>100272／<|begin_of_sentence|>100273／<mask:*>），
  其余 1,009 条占 id 100295..101303 ⇒ 总 id 空间 101,304；
  <0x00>..<0xFF> 256 个字节 token 全在 vocab（id 13..268），byte_fallback 才走得通；
  unk_token=0，ignore_merges=False，dropout=None，pre_tokenizer=null，
  normalizer 只有一条 Replace " "→"▁"，post_processor 是空 TemplateProcessing（不加任何特殊符）。
打包时这几条都重跑一遍断言，不满足就当场停，免得把错口径打进包里。

用法（PC／校机跑，板子上不需要这个脚本）：
    python -X utf8 scripts/mk_e41_tok.py [tokenizer.json 路径] [输出路径]

产物 board/e41_tok.pack；末尾打大小＋md5，登记进台账用。
"""
import hashlib
import json
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON = os.path.join(ROOT, "export", "ernie_fp16_cuda", "tokenizer.json")
OUT = os.path.join(ROOT, "board", "e41_tok.pack")
MAGIC = b"E41TOK01"


def w(u32):
    return struct.pack("<%dI" % len(u32), *u32)


def build(src, dst):
    obj = json.load(open(src, encoding="utf-8"))
    model = obj["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    added = obj["added_tokens"]

    # ---- 断言：口径变了就要重新核，不能悄悄打进包 ----
    assert model["type"] == "BPE", model["type"]
    assert model["ignore_merges"] is False, "ignore_merges 变了：whole-word 捷径要补"
    assert model["dropout"] in (None, 0), model["dropout"]
    assert model["byte_fallback"] is True, "byte_fallback 关了这份包就不成立"
    assert obj["pre_tokenizer"] is None, "有 pre_tokenizer 就得补切词这一层"
    nrm = obj["normalizer"]
    assert nrm["type"] == "Sequence" and len(nrm["normalizers"]) == 1, nrm
    assert nrm["normalizers"][0]["type"] == "Replace", nrm
    assert nrm["normalizers"][0]["pattern"] == {"String": " "}, nrm
    assert nrm["normalizers"][0]["content"] == "▁", nrm
    pp = obj["post_processor"]
    assert pp["type"] == "TemplateProcessing"
    assert not pp["special_tokens"], pp["special_tokens"]
    for a in added:
        assert not a["normalized"] and not a["single_word"] and not a["lstrip"] and not a["rstrip"], a
    ids = sorted(vocab.values())
    assert ids == list(range(len(vocab))), "vocab id 不连续"
    for m in merges:
        assert m[0] in vocab and m[1] in vocab, m
        assert m[0] + m[1] in vocab, m
        assert not m[0].startswith("<0x") and not m[1].startswith("<0x"), m
    seen = set()
    for m in merges:
        k = (vocab[m[0]], vocab[m[1]])
        assert k not in seen, ("重复 merge pair", m)
        seen.add(k)
    byte_ids = [vocab["<0x%02X>" % b] for b in range(256)]

    n_ids = max([max(ids)] + [a["id"] for a in added]) + 1
    assert n_ids < (1 << 17), n_ids
    rank_bits = len(merges).bit_length()
    assert rank_bits + 17 <= 63

    special = sorted(a["id"] for a in added if a["special"])
    tok_list = [None] * len(vocab)
    for k, i in vocab.items():
        tok_list[i] = k
    unk = vocab[model["unk_token"]]

    payload = []
    payload.append(w([n_ids]))
    payload.append(w([unk]))
    payload.append(w([len(special)] + special))
    payload.append(w([len(tok_list)]))
    payload.append(b"".join(struct.pack("<I", len(b)) + b
                            for b in (t.encode("utf-8") for t in tok_list)))
    payload.append(w([len(added)]))
    for a in added:
        eb = a["content"].encode("utf-8")
        payload.append(w([a["id"], len(eb)]))
        payload.append(eb)
    payload.append(w([256] + byte_ids))
    payload.append(w([len(merges), rank_bits]))
    triples = []
    for m in merges:
        triples.append(vocab[m[0]])
        triples.append(vocab[m[1]])
        triples.append(vocab[m[0] + m[1]])
    payload.append(w(triples))
    src_md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    eb = src_md5.encode("ascii")
    payload.append(w([len(eb)]))
    payload.append(eb)
    body = b"".join(payload)
    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF))
        f.write(struct.pack("<I", len(body)))
        f.write(body)
    return dst, src_md5, n_ids, len(tok_list), len(merges), len(added), len(special)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else JSON
    dst = sys.argv[2] if len(sys.argv) > 2 else OUT
    dst, src_md5, n_ids, n_vocab, n_merges, n_added, n_special = build(src, dst)
    raw = open(dst, "rb").read()
    print("源        %s" % src)
    print("src_md5   %s" % src_md5)
    print("词表      %d 条 | id 空间 %d | merges %d | added %d（special %d）"
          % (n_vocab, n_ids, n_merges, n_added, n_special))
    print("包        %s" % dst)
    print("大小      %d B" % len(raw))
    print("md5       %s" % hashlib.md5(raw).hexdigest())


if __name__ == "__main__":
    main()
