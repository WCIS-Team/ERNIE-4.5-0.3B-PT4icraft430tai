# short-kv-window — 编译期 KV 窗口裁剪档（E42=256／E43=128）

这条分支把"把 KV 窗口砍短换速度"那一版做成独立分支，不并进 `master`（现役产品件仍是 E41／NS=512）。
实测数字、判据与公开更正的正本在文档库 `wiki/PL/MOD/llm/编译窗口裁剪实测.md`；本文件只写这条分支装了什么、怎么装、怎么退。

## 一、分支相对 master 多出来的东西

| 件 | 内容 |
|---|---|
| `compile/E42_parse.toml` | 与 `E41_parse.toml` 只差 `net_name`／`jr_path`／`max_tokens_length = 256` |
| `compile/E43_parse.toml` | 同上，`max_tokens_length = 128` |
| `board/e41_chatd.py` | 窄读回抄（＝现役那一版）＋**槽数从图推导**：删掉 `MAX_SLOTS = 511`，改成 `self.max_slots = self.kv_depth - 1`；`ready` 事件回吐 `max_slots`／`kv_depth` |
| `board/e41_chat_console.py` | 删掉 `MAX_SLOTS = 500`，改用 daemon `ready` 里的 `max_slots - 11`（NS=512 时正好还是 500，行为不变） |
| `compile/host_gen.py` | `if past_len >= 512` → `if past_len >= net['depth']`（`depth` ＝ 第一条 past 输入 `shape[2]`） |
| `SHORT_WINDOW.md` | 本文件 |

窄读那一版的探针（`board/e41_narrow.py`／`e41_kvres.py`／`ab_chatd.sh`／`narrow_run.sh`）和它们的读数
`board/narrow/` 已在 `master` 入库（`a9098b1`），本分支直接从它们上面长出来，所以不重复列进上面那张表。
本分支的 `board/e41_chatd.py` ＝ `master` 现役那颗窄读 daemon 只加四处槽数改动（`git diff master..short-kv-window -- board/e41_chatd.py` 只吐出这四处），
所以单独 checkout 这条分支也能跑。

分支基点＝`master` 的 `a9098b1`（早先起在 `9ce054d` 上、那一版分支尖 `8cf5b99` 已作废 ⇒ rebase 过）。
本分支**未推远端**，所以分支尖以现读为准：`git rev-parse --short short-kv-window`。

```sh
cd <本库>          # repos/ERNIE-4.5-0.3B-PT4icraft430tai
git log --oneline master..short-kv-window        # 分支相对 master 的那几笔
git diff --stat master..short-kv-window          # 6 个件
```

件本体（`*.raw` ≈1.0 GB／颗）不进库——单文件 >100 MiB 会被拒推。板上的 E42／E43 在
`/mnt/data/ernie/jr_path/depth/`，登记见 `wiki/PS/板端我方代码与产物台账.md` §卅二。

## 二、为什么槽数必须从图读

`max_tokens_length` 是**编译期**烘进图的 KV 窗口：每条 past 输入的形状是 `[1, 8, NS, 128]` fp16。
`master` 上四处把 512 写死（daemon 511／console 500／host_gen 512／`e41_infer_board.py` 512），
换任何一颗短窗口件都会先在驱动里撞上"槽数比图的深度还大"，`step()` 里那条
`past_len >= self.kv_depth` 直接 `SystemExit`。这条分支把运行时路径上的三处改成读图。

## 三、装（板端）

```sh
# 现役 daemon 先留副本
cp /mnt/data/ernie/e41_chatd.py /mnt/data/ernie/e41_chatd.b512
# 推本分支的 e41_chatd.py 覆盖，然后按件选窗口
E41_JSON=/mnt/data/ernie/jr_path/depth/E42-decoder_ZG.json \
E41_RAW=/mnt/data/ernie/jr_path/depth/E42-decoder_ZG.raw \
E41_SOCK=/tmp/e41_short.sock python3 /mnt/data/ernie/e41_chatd.py serve
```

退：`cp /mnt/data/ernie/e41_chatd.b512 /mnt/data/ernie/e41_chatd.py`，件退回 `jr_path/decode/E41-*`。

## 四、上下文预算（这颗数是这条分支能不能用的判据）

模板本体 ＝ 4 槽；每轮外壳 ＝ user 4 ＋ assistant 4 ＝ 8 槽；中文 ≈0.55 token/字。
客户端裁剪口径是 `len(ids) + 16 <= MAX_SLOTS`，于是"历史＋当轮"的预算：

| NS | daemon `max_slots` | console 预算 | 历史＋当轮可用 ids |
|---|---|---|---|
| 512 | 511 | 500 | ≤484 |
| 256 | 255 | 244 | ≤228 |
| 128 | 127 | 116 | ≤100 |

按陪伴设备的一轮典型（提问 15 字／回答 40 字 ⇒ 该轮进历史后净占 63 槽）：512 存 ≈7 轮、256 存 ≈3 轮、128 存 ≈1 轮。
逐字分解与复现命令在 `workenv/ernie/slot_count.py`（它直接 import 产品自己的 `render()`／`load_tokenizer()`）。

## 五、板上冒烟（2026-11-03 09:30–09:32 UTC，A）

分支 daemon 推成 `/tmp/e41_shortd.py`（**没覆盖现役** `/mnt/data/ernie/e41_chatd.py`），件指 E42：

```
{"ready": true, "apply_s": 45.5, "padfix": true, "max_slots": 255, "kv_depth": 256}
{"end": true, "ids": [5300, 95090, 94179, 11468, 4769, 94042, 94179], "reason": "limit", "sec": 5.2, "prompt": 9}
connect rc=0 wall=122s      # json+raw 54.4 s ＋ apply 45.5 s ＋ 16 步
```

`max_slots`／`kv_depth` 由图推出＝255／256，不是写死的 511／512；16 步、每步 ≈0.29 s（与窗口对照那一轮的 N256 读数 295.38 ms 同量级）。
`E41_IDLE=20` 到点自退，收尾 `pgrep` 零颗。脚本＝`workenv/ernie/board/short_smoke.sh`，读数落点＝板上 `/tmp/short_smoke.txt`。

## 六、这条分支没做的事

- 没换产品件，`master` 一行没动，现役板上跑的仍是 E41／NS=512＋窄读。
- `board/e41_infer_board.py`（10-31 那条单发推理路，`MAX_SLOTS = 512`）与 `eval/` 下几处的 512 常数**未改**：它们不在运行时路径上，改了要重跑那批已归档的读数。
- **多轮真实对话跑穿 256 窗口未测**：上面只冒烟单轮 6 token。回合级要判的是"历史涨到 228 槽以后客户端怎么裁"，那是第五节预算的实测版，本轮未跑。
- 件本体未入库（>100 MiB 拒推），板上 E42／E43 的 md5 与路径在文档库台账 §卅二。
