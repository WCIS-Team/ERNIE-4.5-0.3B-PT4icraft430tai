# ERNIE-4.5-0.3B-PT4icraft430tai · 迁移交付件

> 这条线的仓：把 ModelScope `PaddlePaddle/ERNIE-4.5-0.3B-PT`（18 层／hidden 1024／16 Q＋2 KV head／head_dim 128／vocab 103,424，参数 **360,748,032 ≈ 0.36 B**，A：safetensors 头部 164 张量逐张求和、全 BF16）推到 iCraft（30TAI／ZG330）的 `_ZG` 件，并把它在板外的两条主机上量完，最后在板上点火。**2026-10-30 立项**（她那句"砍掉 Q3I 那两份 1.6 GB，然后开 ERNIE 这条线"），**10-30～10-31 走完门 1～3＋PC 数值＋校机实测，11-01 上板走完门 4**。
> **门 4 已通（2026-11-01）**：板上 `precheck=True`、逐 token 与 PC Host 后端对上（padfix 两模式各 7～8 步）、稳态 **0.42 s/步＝2.38 token/s**（这是驱动的 `dt` 口径，**不含读回与 argmax**；轮级墙钟 0.47 s/步，差 46.2 ms，拆开见「优化空间判定」）、64 token 长跑到 eos 无泄漏；padfix 板上 A/B 后**默认开**。**可复现的是板端**（两次跑 11 步 logits 极值逐位相同）；**PC Host 后端复跑会抖**（同一 42 路输入 |Δ| 均值 0.177／最大 0.668，并列第一会翻 argmax），所以"全同"要带 top-2 间距读。板上件与读数＝`wiki/PS/板端我方代码与产物台账.md` §廿五（#218～#222）。
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
| `board/` | 板端 5 件＋4 份日志：`e41_probe.py`（`E41STEP=0..3` 分步探针，量 `VmRSS/VmHWM`＋`MemAvailable/CmaFree`＋logic/phy 段）；`e41_infer_board.py`（常驻驱动：42 路位置式喂入、present 整块回喂、`E41_PADFIX` 首步去毒、`E41_STOP` 默认 2）；**11-01 新增 `e41_chatd.py`**（对话守护：模型只 apply 一次，unix socket 收 `{"ids":…,"max_new":N}`，空闲 300 s 自退）；**`e41_chat_console.py`**（本机控制台客户端，桌面上那份 `ERNIE对话.py` 与它逐字节相同）；**11-01 晚新增 `e41_prof.py`**（`E41P_MODE=wall|prof` 六段计时＋read 成本电池，即「优化空间判定」那两轮探针）。日志 `e41p_wall.txt`／`e41p_prof.txt`＋两张 ops CSV 是板上现读原件取回，两端 md5 一致。**读数是板上现读**，正文在 wiki 台账 §廿五＋§廿六＋§廿八 | 是 |
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

## 板端对话入口（2026-11-01 晚）

分两半：**本机** `e41_chat_console.py`（＝桌面上那份 `ERNIE对话.py`）做分词／解码／交互；**板上** `e41_chatd.py` 常驻，模型只装一次。

- **用法**：双击桌面 `ERNIE对话.py`。命令 `/clear` 清上下文、`/max N` 改生成长度（默认 96）、`/stats` 看上一轮速度、`/exit` 退出。
- **冷启动 ≈ 106～110 s**（11-01 四次墙钟：`json+raw` 50.8～52.2 s ＋ 建映射／开设备／建会话 11.5～12.9 s ＋ `apply` 43.5～45.8 s，都是板上现读；旧写"95 s"漏了中间那段）；服务已在时秒连。每轮把整段对话重放一遍（不是增量续 KV），所以**轮数越多越慢**，超过 500 槽自动丢最早一轮。
- **实测**：37 token／23.4 s（含 13 步 prefill，合计 50 步 ⇒ 0.47 s/步）、38 token／25.2 s（含 16 步 ⇒ 0.47 s/步），两轮都以 eos 收尾，输出中文通顺。多轮第二问：prompt 123 步＋97 token＝220 步／102.4 s，仍是 0.47 s/步。
- **默认 96 token 上限容易撞**：这颗模型遇到"1+1 等于几"也会展开成长篇，撞上限时以 `limit` 截断（不是 eos）；要它说完就 `/max 160`。
- **为什么分词在本机**：板上 `tokenizers` 0.13.3 读不了这颗 `tokenizer.json`（新版 `ignore_merges` 字段），守护只收 id、回 id；控制台按 `chat_template.jinja` 手工渲染（tokenizers 0.23.3 没有 `apply_chat_template`）。
- **板子占用**：客户端连上时把 `deb-h61:/home/user/Desktop/113.txt` 写成 `test==on`，退出时恢复原值；守护在最后一个客户端断开后**空闲 300 s 自退**（想立刻放开就 `ssh root@192.168.1.113 "pkill -f 'e41_chatd[.]py serve'"`，方括号是防自杀，见台账 §廿六）。

## 优化空间判定（2026-11-01，A＋B 两轮板上实测）

起因是审厂商参考工程 `repos/official/fpai_reference_design` 的 skill 后留下的问题：**"这颗件的 0.47 s/步还有多少可挤？"** 用 `board/e41_prof.py` 上板量了两轮（同进程、`ocm=1`／`padfix=1`、板钟 12:13–12:19 UTC）。**读数正本＝`wiki/PS/板端我方代码与产物台账.md` §廿八**，这里只写判定和口径。

### 先更正一条：0.42 与 0.47 不是两台东西在打架

`e41_chatd.py:144` 的 `dt = time.time() - t0` 打在 `forward`＋`waitForReady`＋`reset` 之后、**所有 `Tensor.read` 之前**；§25.3 的 **0.42 s/步** 是这个 `dt`，§26.1／本页的 **0.47 s/步** 是整轮墙钟÷步数。同场六段中位（ms）：

| `forward` | `waitForReady` | `reset` | `read_logits` | `read_kv36` | `argmax` | **轮级** |
| --- | --- | --- | --- | --- | --- | --- |
| 393.13 | 25.41 | 0.02 | 0.32 | **37.76** | 8.12 | **464.83** |

⇒ `dt` 口径 418.542 ms、墙钟 464.828 ms、**差＝读回尾巴 46.203 ms**（＝任何"只动读回"的优化的天花板）。六段各取中位，相加 464.76 与轮级中位 464.83 差 0.07 ms＝中位数不可加，不是漏项。原先把这两个数当"不同负载下的两次测量"是**不准确的，现在按计时点差异归因结案（A）**。

### 再算清窄读：成立，但只值 5.6 %

把 36 条 present 从"整块 1 MiB 读回"改成"只读本步新槽"，字节数 **36 MiB → 72 KiB**（512× 看着很诱人）。但 KV 轴序是 `[batch, head, slot, head_dim]`，**一步新写的 2 KiB 在内存里就是 8 段各 256 B**（head 间隔 512×128×2＝131,072 B），所以窄读＝**288 次 256 B 碎片调用**。板上电池实测（`REPS=5` 取 min）：

| 读法 | 次数 | 字节 | ms |
| --- | --- | --- | --- |
| 整块（现役） | 36 | 36 MiB | **35.18** |
| 真窄读切片 | **288** | 72 KiB | **9.11** |
| 连续 2 KiB 行（若轴序是 slot-major） | 36 | 72 KiB | **1.26** |
| PS 内存 memcpy（基线） | 36 | 36 MiB | 12.43 |

按 `T = a·calls + r·bytes` 两点联立解出 **a＝31.4 µs/次、r＝1108.7 MB/s**，第三档预测 1.20 vs 实测 1.26 ms（+5 %）⇒ 模型站得住。**净省 35.18 − 9.11 ＝ 26.07 ms/步＝轮级 5.6 %**（464.83 → 438.76 ms，2.15 → 2.28 token/s）；盈亏平衡要单次开销 > 122.1 µs，实测只有 31.4 µs ⇒ **方向对，但 46.2 ms 的天花板被 288 次调用本身吃掉 20 ms**。

### 判定

| 项 | 改动 | 收益 | 结论 |
| --- | --- | --- | --- |
| **C** 窄读 | 只改 `board/e41_chatd.py` 的读回，不重编 | **26.07 ms/步（5.6 %）** | **成立但不厚。未实装**——她 11-01 只批了 A/B，C 要她点头才上板 |
| **D** slot-major 重编 | 把 cache 轴序换成 `[batch, slot, head, head_dim]`，让新槽连续 | 在 C 之上再 **9.11 − 1.26 ＝ 7.85 ms（1.7 %）**；C＋D 合计 33.92 ms（7.3 %） | **放着不做**（她 11-01 裁）：为 1.7 % 重走五段编译＋重新数值对齐不值 |

### 顺带查清的四条（都影响后续优化选型）

- **冷启动那 44 s `apply` 省不掉**：官方 `docs/demo/samples/xrt/snapshot.html` 明写"目前序列化 session 仅支持 **BY** 阶段的网络"⇒ ZG 件没有快照复用这条路（A，官方文档级）。
- **`precheck()` 自己就要 43 s** ⇒ 只能当一次性门禁，**绝不能进热路径**（A）。
- **`mmuMode` 参数在 ZG 分支是空转**：厂商 helper 只在 buyi 分支调 `mmuModeSwitch`（`pyrtutils/icraft_utils.py:105`）（A，源码直读）。
- **profiler 的占比**：`timeProfileResults()` 只有 **62 条**（不是 1373 个算子，疑合并后的执行单元，**归属未取证**），一次前向 total 399.1 ms ＝ memcpy **118.1（29.6 %）**／hard **114.1（28.6 %）**／other **166.9（41.8 %）**；两次 dump（1 步后 vs 7 步后）SUM 几乎不变 ⇒ **per-forward 语义**。top memcpy 是 `op_id 9930` ＝ 58.843 ms（占 memcpy 49.8 %），**与"每步 36 MiB 上行 ≈ 642 MB/s"量级吻合，但认成 KV 上行只是推断（C）**——CSV 的 `op_id` 与图 JSON 的 `op_id` 零命中。**最大的一块是 `other` 41.8 %，拆它要的是 op 级归属，窄读动不到 forward 内部。**
- **仍没跑的**：⑤ ocm `BEST_SCORE` vs 现役 `OPTION1` 的 A/B——Python helper 表达不了 `-1`（`ocmOptimize(OcmOpt.BEST_SCORE)` 得直接对 backend 调），要同进程换档，探针没做这层。

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
| 板端对话入口（11-01 晚） | **通**（A：冷启动拉起→ready→两轮问答均到 eos；热连接秒回；真实控制台窗口中文渲染正常） |
| 每步耗时拆开＋窄读收益判定（11-01 夜，A＋B） | **通**（A：轮级 464.83 ms／`dt` 口径 418.54 ms／读回尾巴 46.20 ms；窄读净省 26.07 ms＝**5.6 %** ⇒ **C 成立、未实装**；D 上限再 7.85 ms＝**1.7 %** ⇒ **她裁"放着不做"**）；profiler 占比 memcpy 29.6 %／hard 28.6 %／other 41.8 %（**op 归属未取证**）。台账 §廿八 #229 |
| 板上直接吃中文（分词在板上做） | **未执行**：板上 `tokenizers` 0.13.3 读不了 ERNIE 的 `tokenizer.json`（新版 `ignore_merges` 字段），驱动现为"喂 id／出 id" |
| 量化档（int8／mix）、25 token 长链逐级比对、英文退化归因 | **未执行** |
