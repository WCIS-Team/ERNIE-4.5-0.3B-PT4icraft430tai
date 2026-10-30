# ERNIE-4.5-0.3B-PT4icraft430tai · 迁移交付件

> 这条线的仓：把 ModelScope `PaddlePaddle/ERNIE-4.5-0.3B-PT`（18 层／hidden 1024／16 Q＋2 KV head／head_dim 128／vocab 103,424，参数 **360,748,032 ≈ 0.36 B**，A：safetensors 头部 164 张量逐张求和、全 BF16）推到 iCraft（30TAI／ZG330）的 `_ZG` 件，并把它在板外的两条主机上量完。**2026-10-30 立项**（她那句"砍掉 Q3I 那两份 1.6 GB，然后开 ERNIE 这条线"），**10-30～10-31 走完门 1～3＋PC 数值＋校机实测**。
> **门 4 零读数**：`.raw`／`.json` 已在盘，但板上没传、没点火——按她的要求，**上板前先跟她说**。
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
- **修法＝驱动约定**：token 0 连喂两次＋每步把槽 1 复制进槽 0（`host_gen.py --padfix`）。替身面 120 题净收益（裸提示 `top1==HF` 23→40、chat 面 11→56）；编译件面已在 10-31 复核（8 步，三条控制全过，首 token 与 HF 同为 23）。**板上默认开不开还没定**（`pipeline.md` §12 第 3 条）。

## 状态（10-31）

| 步 | 状态 |
|---|---|
| 权重下载（校机，ModelScope 官方工具） | **通**（A） |
| genai 导出 `-p fp16 -e cuda` | **通**，EXIT=0（A） |
| 图清点 ＋ 与 Qwen3 对照 | **通**（A） |
| 门 0：HF fp32 基准（prefill＋1 步 decode） | **通**（A：PREFILL cos 0.999999／top5 逐字相同） |
| 门 1 前端摄取（ir／opset／符号轴／qkv／几何） | **通**（A，十三步改写链，`e_v41`） |
| 门 2 图合同（42 条输入、`token_dynamic_fm @fp(32)[1,3]`、decode-only 签名） | **通**（A，`pipeline.md` §8.1 合同已写死；**板端驱动代码未写**） |
| 门 3 算子归属 | **通**（A：1332 颗 `@zhuget(330)`／41 颗 `@hostt`，host 侧全是动态控制类，非本颗独有缺口） |
| PC 侧数值：本机 Host 后端逐级链 | **通**（A：7/7 argmax 与替身逐位相同；首步除外＝已知缺陷） |
| 校机实测 60 题＋chat 面＋长度／困惑度扫描 | **通**（A：`pipeline.md` §9；可用性＝中文窄可用、英文链明显退化） |
| padfix（slot1→slot0）复核 | **替身面通、编译件面通**（A，§9.6／§9.7）；60 题编译件面**未跑** |
| 门 4 板端（CMA／位流门禁、传输、驱动、点火） | **未执行**；动板前先查 `113.txt`，且先跟她说 |
| 量化档（int8／mix）、25 token 长链逐级比对、英文退化归因 | **未执行** |
