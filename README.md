# ERNIE-4.5-0.3B-PT4icraft430tai · 迁移交付件

> 这条线的仓：把 ModelScope `PaddlePaddle/ERNIE-4.5-0.3B-PT`（18 层／hidden 1024／16 Q＋2 KV head／head_dim 128／vocab 103,424，参数 **360,748,032 ≈ 0.36 B**，A：safetensors 头部 164 张量逐张求和、全 BF16）推到 iCraft（30TAI／ZG330）的 `_ZG` 件，并把它在板外的两条主机上量完，最后在板上点火。**2026-10-30 立项**（她那句"砍掉 Q3I 那两份 1.6 GB，然后开 ERNIE 这条线"），**10-30～10-31 走完门 1～3＋PC 数值＋校机实测，11-01 上板走完门 4**。
> **门 4 已通（2026-11-01）**：板上 `precheck=True`、逐 token 与 PC Host 后端对上（padfix 两模式各 7～8 步）、稳态 **0.42 s/步＝2.38 token/s**、64 token 长跑到 eos 无泄漏；padfix 板上 A/B 后**默认开**。**可复现的是板端**（两次跑 11 步 logits 极值逐位相同）；**PC Host 后端复跑会抖**（同一 42 路输入 |Δ| 均值 0.177／最大 0.668，并列第一会翻 argmax），所以"全同"要带 top-2 间距读。板上件与读数＝`wiki/PS/板端我方代码与产物台账.md` §廿五（#218～#222）。
> 全部读数与流程＝本目录 [`pipeline.md`](pipeline.md)（正本，含 §10 四条公开更正）；介绍页＝`wiki/PL/MOD/llm/ERNIE-4.5-0.3B-PT.md`；四道门判据正本＝`wiki/PL/MOD/llm/LLM上板要求与候选模型对照.md`；配方来源＝[[Qwen3-Instruct按Qwen3配方迁移]] 与 `repos/Qwen3-0.6B-Instruct/pipeline.md`。**本页只做索引与状态，不复制读数表。**
> 与 vpu-live／Qwen3 同模式：小件与证据进 git、体积件挡在库外（`.gitignore` 的 `*.onnx`、`*.onnx.data`、`*.raw`、`*.safetensors`、压缩包等）。

## 布局

| 子目录／件 | 内容 | 入库 |
|---|---|---|
| `pipeline.md` | 流程正本，13 节：结论／主机分工／导出／图清点／门 0 HF 基准／`interleaved` 来龙去脉／**门 1 十三步改写链**／**门 2＋3 五段与三笔债**／PC 侧 `feed` 合同与逐级链／校机实测 9.1～9.7／更正记录／复跑／还没做／路径对照。读数带证据分级（A/B/C/D） | 是 |
| `export/ernie_fp16_cuda/` | genai fp16/cuda 导出件里的**小件 9 份**（`chat_template.jinja`／`config.json`／`genai_config.json`／`generation_config.json`／`added_tokens.json`／`special_tokens_map.json`／`tokenizer.json`／`tokenizer.model`／`tokenizer_config.json`），逐件 md5 与校机相等 | 是 |
| 同上 `model.onnx`＋`.data` | 103,511 B／`6f07fa16…` ＋ 755,105,792 B／`788afdcb…`，**正本留校机**，按"本机不下 GB 级资源"没往回拷 | 否（体积件） |
| `compile/` | **门 1～3 现役脚本 58 件 ＋ 现役 toml `E41_parse.toml`（902 B／`00a4ce9f…`）**：13 步改写链（`meta_variant`→`bake_kv_dim`→`split_qkv`→`constantize_split_sizes`→`shrink_rope`→`patch_attr`→`tile_kv_heads`→`reshape_gqa_inputs`→`decouple_gqa_inputs`→`add_head_roundtrip`→`crop_layers`→`pad_cache_inputs`→`permute_rope_heads`）＋控制实验与清点（`zg_census`／`cmp_graph_context`／`mk_minimal_gqa`／`roundtrip_control`…）＋`host_gen.py`（本机 Host 后端逐级链驱动，带 `--padfix`）＋`mk_toml.py`／`mk_ort_variant.py` | 是 |
| `eval/` | 校机实测：**脚本** `eval_ernie.py`（五列对照＋`--mode check` 替身证明）、`eval_padfix_compiled.py`（编译件打分）、`probe_e41_ort.py`；**读数** `eval_items{,_chat}.json`（60 题×5 列）、`sweep.json`（长度＋困惑度）、`pad.json`（`SLOTS` 五档）、`cachemap.json`（5 例×18 层基座对照）、`decomp.json`（误差分解）、`check_e41.json`（替身⇄编译件 7 步）、`padfix{,_chat}.json`（修复面 60＋60）、`items.json`（题库） | 是 |
| `eval/compiled_chain/`、`eval/compiled_padfix/` | 本机编译件逐级链的**每步 logits**（`step<N>/logits.npy`，fp32 103,424 元素＋128 B 头＝413,824 B／件）＋`steps.json`／`tokens.txt`；`smoke_2token/`＝`--padfix` 索引越界修好后的 5 步复跑证据 | 是 |
| `logs/` | 门 0 三份读数：`logs_export.txt`（41 行，末尾 `EXIT=0`）、`logs_census.txt`（ERNIE×Qwen3 图清点）、`logs_verify.txt`（HF 基准逐步）。UTF-16LE 原件与逐里程碑日志（parse/optimize/quantize/adapt/generate）在 `workenv\ernie\`，不入库 | 是 |
| `scripts/` | 校机侧现役 10 件：`dl_ernie.py`（ModelScope 官方工具）、`export_a.ps1`、`census_onnx.py`、`verify_onnx_ernie.py`、`ver_probe.py`、`probe_{genai_ernie,ernie_class,ernie_rope,interleaved_use,ernie_files}.py` | 是 |
| `board/` | 板端 2 件（10-31 写、11-01 上板跑通）：`e41_probe.py`（`E41STEP=0..3` 分步探针，量 `VmRSS/VmHWM`＋`MemAvailable/CmaFree`＋logic/phy 段）；`e41_infer_board.py`（常驻驱动：42 路位置式喂入、present 整块回喂、`E41_PADFIX` 首步去毒、`E41_STOP` 默认 2）。**读数是板上现读**，正文在 wiki 台账 §廿五 | 是 |
| `MANIFEST.md5` | 全目录 md5 清单（vpu-live 姿势） | 是 |

**五段中间件与 `fmodel/e_v1`～`e_v41` 不在本仓**（体积、可复跑）：正本在编译根 `S:\apps\llm\agentwork\ernie\compile\`，`E41-decoder_ZG.json` 1,637,490 B／`E41-decoder_ZG.raw` 1,054,488,148 B／`params_md5` `6c0fe5afb0d744a9dfdd20c5959b1007`（＝`.raw` 的 md5，逐字节相等）。要复算某一步改写回编译根取。

## 这颗件绑什么——"能不能换个 NPU 用"

一句话：**绑"器件档"（Zhuge 核 ZG330），不绑这块板**；换架构不可能，同档换板**没测过**。论证与出处（`compile_target.h` 的枚举、3.39.0 的设备运行时清单、官方算子表按器件分列）写在对照页 **§九**，此处不复制。

## 三条为迁就 parser 付的债（详情＝`pipeline.md` §7）

1. **KV head 2→8 平铺**：ERNIE 原生 2 个 KV head，parser 只认得 Qwen3 那套几何 ⇒ 每颗 cache `[1,8,512,128]` fp16＝1,048,576 B，是原生 2 头形态的 4 倍。**上板预算按 8 头算。**
2. **补 4 条没人消费的 cache 声明**（38→42）：parser 按**声明条数**注册 KV 输入（≤40 挂、≥42 过），与层数无关 ⇒ 每步多喂 4 MiB 死数据，板端驱动必须喂满 42 路。
3. **RoPE 配对改数据不改属性**：parser 把 `rotary_interleaved` 整个丢掉（两份 IR 只差属性时 logits 逐比特相同）⇒ 只能按 `PERM=[0,2,…,127,1,3,…]` 置换 qkv 权重的 head_dim 轴。

## 已知缺陷与修法

- **首步（位置 0）坏**：喂进去的第 0 槽是全零 pad，而图真的会去 attend 它 ⇒ 不修时首步 `comp_vs_hf` cos **−0.253818**；缩短 cache（`SLOTS` 512→16）读数逐位不变 ⇒ **不用重编**。
- **修法＝驱动约定**：token 0 连喂两次＋每步把槽 1 复制进槽 0（`host_gen.py --padfix`）。替身面 120 题净收益（裸提示 `top1==HF` 23→40、chat 面 11→56）；编译件面已在 10-31 复核（8 步，三条控制全过，首 token 与 HF 同为 23）。
- **板上默认开（11-01 A/B 定）**：不修时同一提示退化成时间戳循环（`\n00:00:00.000 - …`）；修后 55 token 通顺切题到 eos。代价＝每轮多一次前向，实测速度无可测差（0.42 vs 0.42 s/步）。

## 状态（2026-11-01）

| 步 | 状态 |
|---|---|
| 权重下载（校机，ModelScope 官方工具） | **通**（A） |
| genai 导出 `-p fp16 -e cuda` | **通**，EXIT=0（A） |
| 图清点 ＋ 与 Qwen3 对照 | **通**（A） |
| 门 0：HF fp32 基准（prefill＋1 步 decode） | **通**（A：PREFILL cos 0.999999／top5 逐字相同） |
| 门 1 前端摄取（ir／opset／符号轴／qkv／几何） | **通**（A，十三步改写链，`e_v41`） |
| 门 2 图合同（42 条输入、`token_dynamic_fm @fp(32)[1,3]`、decode-only 签名） | **通**（A，`pipeline.md` §8.1；板端驱动 `board/e41_infer_board.py` 按此合同写并上板跑通） |
| 门 3 算子归属 | **通**（A：1332 颗 `@zhuget(330)`／41 颗 `@hostt`，host 侧全是动态控制类，非本颗独有缺口） |
| PC 侧数值：本机 Host 后端逐级链 | **通**（A：7/7 argmax 与替身逐位相同；首步除外＝已知缺陷） |
| 校机实测 60 题＋chat 面＋长度／困惑度扫描 | **通**（A：`pipeline.md` §9；可用性＝中文窄可用、英文链明显退化） |
| padfix（slot1→slot0）复核 | **替身面通、编译件面通**（A，§9.6／§9.7）；60 题编译件面**未跑** |
| 门 4 板端（CMA／位流门禁、传输、驱动、点火） | **通**（A，11-01：`precheck=True`／`apply` 46.4 s／logic 936.6 MiB、WEIGHT 742.2 MiB 落 `plddr`／CMA 起跑前 ≥126 MiB／`VmHWM` 505 MB 峰值）；**逐 token 与 PC 对上**（padfix 0／1 各 7／8 步；间距 <0.7 的步不作硬判据），稳态 0.42 s/步，62 步到 eos 无泄漏。**板端两次跑逐位可复现；PC Host 后端复跑有 0.177～0.668 抖动**。台账 §廿五 #218～#222 |
| padfix 板上 A/B | **该开，已定为默认**（A：关→时间戳循环退化；开→55 token 通顺到 eos；速度无可测差） |
| 板上直接吃中文（分词在板上做） | **未执行**：板上 `tokenizers` 0.13.3 读不了 ERNIE 的 `tokenizer.json`（新版 `ignore_merges` 字段），驱动现为"喂 id／出 id" |
| 量化档（int8／mix）、25 token 长链逐级比对、英文退化归因 | **未执行** |
