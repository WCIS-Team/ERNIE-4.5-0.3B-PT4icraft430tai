#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板侧分词 board/e41_tok.py 对参照物的逐条对照：ids 与解码串都必须全等，一条不一致就非零退出。

参照物 = 本机 tokenizers 读那颗 tokenizer.json（与 board/e41_chat_console.py 用的是同一颗件、
同一条 API），不是"看起来对得上"，是逐条比：
    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(JSON).encode(text).ids          # 参照
    got = e41_tok.load().encode(text)                          # 板侧
覆盖面（--quick 只跑前四类，全量约几分钟）：
    A 真实文本：仓里现成的板端 prompt／评测条目／文档，整份一次＋逐行一次
    B 现役对话模板：board/e41_chat_console.py 的 render() 那种拼法，多轮若干组合
    C 词表自噬：100,295 条词条逐条当输入编码（专打 merge 边界与 added token 抢位）
    D 全部 added token：1,029 条，另加前缀互掐的 <|LOC_1|>／<|LOC_10|> 这类
    E 随机 fuzz：加权字符池（CJK／ASCII／数字／标点／空格／U+2581／控制符／emoji／生僻码位），定 seed
    F 对抗拼接：随机词条串接＋往缝里插特殊符与数字
    G 解码全表：id 0..101303 逐条解码，另加随机 id 组（skip_special_tokens 两档都跑）
    H 往返：上面每条文本 encode→decode 再比一次
    I 口径核对：add_special_tokens 真／假在这份 tokenizer 上是否真的等价（板侧只有一条路径，
      这条不成立就得补分支）

用法（PC／校机，板子上没有 tokenizers 跑不了本件）：
    python -X utf8 scripts/tok_parity.py [--quick] [条数倍率]
"""
import io
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON = os.path.join(ROOT, "export", "ernie_fp16_cuda", "tokenizer.json")
sys.path.insert(0, os.path.join(ROOT, "board"))

try:
    from tokenizers import Tokenizer
    import tokenizers
except ImportError:
    sys.exit("缺 tokenizers，本件是 PC 侧对照器：pip install tokenizers 或换本机跑")

import e41_tok

BOS = "<|begin_of_sentence|>"
EOS = "<|end_of_sentence|>"
REAL = ["board/narrow/PB.txt", "board/narrow/PJ.txt", "eval/items.log",
        "board/oracle/best.json", "export/ernie_fp16_cuda/tokenizer_config.json",
        "export/ernie_fp16_cuda/special_tokens_map.json", "README.md", "pipeline.md",
        "board/e41_chat_console.py", "board/e41_chatd.py", "board/e41_tok.py",
        "MANIFEST.md5"]


def render(history):
    """board/e41_chat_console.py:46 那套模板，逐字照搬。"""
    s = BOS
    for m in history:
        if m["role"] == "user":
            s += "User: " + m["content"] + "\n"
        elif m["role"] == "assistant":
            s += "Assistant: " + m["content"] + EOS
        else:
            s += m["content"] + "\n"
    return s + "Assistant: "


def texts_real():
    out = []
    for rel in REAL:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            continue
        body = io.open(p, encoding="utf-8", errors="replace").read()
        out.append(body)
        out.extend(body.splitlines())
    return [t for t in out if t]


def texts_template(rnd, n):
    hi = ["你好，请问悟净开发板的 NPU 有多大内存？", "把这段中文翻成英文：今天有风。",
          "1+1=? 只回数字", "帮我写一个 C 语言 hello world", "为什么 apply 要 90 秒？",
          "Mixed 中英 code `print(1)` 与 2026 年", "　全角空格与\t制表\n换行\r回车",
          "emoji 😀🎉 与生僻 鿿龘", "  多空格   前 后  ", EOS + BOS, "<unk>",
          "".join(chr(rnd.randrange(0x4e00, 0x9fff)) for _ in range(120))]
    out = []
    for k in range(1, 7):
        for i in range(0, len(hi), 2):
            h = [{"role": "user", "content": hi[j % len(hi)]} for j in range(k)]
            out.append(render(h))
    for _ in range(n):
        m = []
        for t in range(rnd.randrange(1, 6)):
            m.append({"role": "user" if t % 2 == 0 else "assistant",
                      "content": rnd.choice(hi)})
        out.append(render(m))
    return out


def texts_vocab(ref, limit):
    return [ref.id_to_token(i) for i in range(min(limit, ref.get_vocab_size()))]


def texts_added(ref, rnd, n):
    sp = [t for t in (ref.id_to_token(i) for i in range(101304))
          if t is not None and t.startswith("<|")]
    out = list(sp)
    out += ["<|LOC_1|>", "<|LOC_10|>", "<|LOC_1|><|LOC_10|>", "前<|LOC_1|>后<|LOC_10|>尾",
             "<|begin_of_sentence|><|end_of_sentence|>", "<|LOC_", "<|LOC", "<<|LOC_1|>>",
             "<unk>x<s>", BOS + "2026" + EOS]
    for _ in range(n):
        k = rnd.sample(sp, min(len(sp), rnd.randrange(1, 5)))
        out.append("".join(k) if rnd.random() < .5 else " ".join(k))
    return out


def texts_fuzz(rnd, n, maxlen):
    pool = ("".join(chr(c) for c in range(32, 127)) * 3
            + "你好我是中国人开发板内存编译模型服务接口参数温度权重推理" * 6
            + "0123456789" * 4 + " \t\n\r" * 5 + "▁▁▁"
            + "，。？！：；、“”（）《》" * 3 + "áéñüÖß" * 2 + "ΩμπσΔ" * 2
            + "😀🎉🚀🐍" + "鿿龘彞" + "́​‿"
            + "<|>" * 8 + "ABCDEFabcdef" * 2 + "　 、" + chr(0) + chr(1) + chr(2))
    out = []
    for _ in range(n):
        L = rnd.randrange(1, maxlen)
        out.append("".join(rnd.choice(pool) for _ in range(L)))
    return out


def texts_adversarial(ref, rnd, n):
    toks = [ref.id_to_token(i) for i in rnd.sample(range(ref.get_vocab_size()), 400)]
    out = []
    for _ in range(n):
        k = rnd.randrange(1, 14)
        parts = [rnd.choice(toks) for _ in range(k)]
        glue = rnd.choice(["", " ", BOS, EOS, "0", "<unk>", "▁", "\n"])
        out.append(glue.join(parts))
    return out


def id_groups(ref, rnd, n):
    vs = ref.get_vocab_size()
    out = []
    for _ in range(n):
        out.append([rnd.randrange(0, 101304) for _ in range(rnd.randrange(1, 40))])
    out.append(list(range(vs)))
    out.append([100273, 104, 2, 100295, 101303, 253, 172, 165, 141])
    return out


def main(argv):
    quick = "--quick" in argv
    argv = [a for a in argv if a != "--quick"]
    mult = int(argv[0]) if argv else 1
    rnd = random.Random(20261104)
    ref = Tokenizer.from_file(JSON)
    mine = e41_tok.load()
    print("参照物 tokenizers %s | 板侧包 %s" % (tokenizers.__version__,
                                                os.path.basename(mine.pack)))
    stats = []
    bad = []

    def case(name, texts):
        t0 = time.time()
        n = 0
        for t in texts:
            n += 1
            a = ref.encode(t).ids
            b = mine.encode(t)
            if a != b:
                bad.append(("encode", name, t, a, b))
        stats.append((name, n, round(time.time() - t0, 2)))

    case("A 真实文本", texts_real())
    case("B 现役模板", texts_template(rnd, 60 * mult if not quick else 12))
    if not quick:
        case("C 词表自噬", texts_vocab(ref, 100295))
    else:
        case("C 词表自噬", texts_vocab(ref, 1500))
    case("D added token", texts_added(ref, rnd, 200 * mult if not quick else 30))
    case("E 随机 fuzz", texts_fuzz(rnd, 4000 * mult if not quick else 300, 240))
    case("F 对抗拼接", texts_adversarial(ref, rnd, 3000 * mult if not quick else 200))

    t0 = time.time()
    nd = 0
    for tid in range(101304):
        nd += 1
        if ref.decode([tid]) != mine.decode([tid]):
            bad.append(("decode1", "G 解码全表", tid, ref.decode([tid]), mine.decode([tid])))
    for grp in id_groups(ref, rnd, 300 * mult if not quick else 40):
        for skip in (True, False):
            nd += 1
            a = ref.decode(grp, skip_special_tokens=skip)
            b = mine.decode(grp, skip_special_tokens=skip)
            if a != b:
                bad.append(("decodeN", "G 解码全表", grp[:12], a, b))
    stats.append(("G 解码全表", nd, round(time.time() - t0, 2)))

    t0 = time.time()
    nr = 0
    for name, texts in [("A", texts_real()), ("H fuzz", texts_fuzz(rnd, 600 * mult, 200))]:
        for t in texts:
            nr += 1
            if ref.decode(ref.encode(t).ids) != mine.decode(mine.encode(t)):
                bad.append(("roundtrip", name, t[:60], "", ""))
    stats.append(("H 往返", nr, round(time.time() - t0, 2)))

    t0 = time.time()
    ni = 0
    sample = texts_fuzz(rnd, 400, 120) + texts_real()[:200]
    for t in sample:
        ni += 1
        if ref.encode(t, add_special_tokens=True).ids != ref.encode(t, add_special_tokens=False).ids:
            bad.append(("add_special", "I 口径", t[:60], "", ""))
    stats.append(("I add_special 等价", ni, round(time.time() - t0, 2)))

    print("%-18s %10s %8s" % ("档", "条数", "秒"))
    for name, n, dt in stats:
        print("%-18s %10d %8s" % (name, n, dt))
    print("合计 %d 条比对，不一致 %d 条" % (sum(s[1] for s in stats), len(bad)))
    for kind, name, t, a, b in bad[:10]:
        print("--- %s / %s" % (kind, name))
        print("    输入 %r" % (repr(t)[:300],))
        print("    参照 %r" % (repr(a)[:300],))
        print("    板侧 %r" % (repr(b)[:300],))
        if isinstance(a, str) and isinstance(b, str):
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            print("    首差位置 %d | 参照 %r | 板侧 %r"
                  % (k, a[max(0, k - 20):k + 20], b[max(0, k - 20):k + 20]))
    if len(bad) > 10:
        print("    …另有 %d 条" % (len(bad) - 10,))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
