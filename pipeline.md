# ERNIE-4.5-0.3B-PT · NPU 迁移流程正本（导出→改写→五段编译→PC 校验→校机实测→板上点火）

> 立项 2026-10-30，按她那句"开 ERNIE 这条线（开的话第一步只是'校机导出＋对 HF 基准'，在 WCIS/repos 里开一个 ERNIE 的 repo）"。**10-30 当天走到的只是第一步**；10-30 晚～10-31 凌晨把门 1～3、PC 侧数值校验和校机实测一并跑完，**11-01 上板把门 4 走通**（§14），本页现在记的是**整条链**，第一步那批读数留在 §2～§4。
> 上板判据的正本＝`wiki/PL/MOD/llm/LLM上板要求与候选模型对照.md`（四道门那张表在 §二）；本条线的"介绍页"＝`wiki/PL/MOD/llm/ERNIE-4.5-0.3B-PT.md`（只写型号与状态，读数指回本页，不复制表格）；配方来源＝[[Qwen3-Instruct按Qwen3配方迁移]] 与 `repos/Qwen3-0.6B-Instruct/pipeline.md`。
> **证据分级**：A＝从件／日志／终端直取；B＝原文照录或推导；C＝按比例外推、未实测；D＝听说。**没执行的步骤一律写成"未执行"**，不写成结果。§10 单列我判错过又更正的三条。

## 0. 结论（读数分别在括号里）

1. **导出通**（A，§2）：`onnxruntime_genai.models.builder` 吃 `PaddlePaddle/ERNIE-4.5-0.3B-PT`，`-p fp16 -e cuda`，`EXIT=0`，产物 6 件／766,418,358 B。
2. **门 1～3 通，产物在盘**（A，§6～§7）：`E41-decoder_ZG.json` 1,637,490 B／`E41-decoder_ZG.raw` 1,054,488,148 B，`params_md5` ＝ `.raw` 的 md5 ＝ `6c0fe5afb0d744a9dfdd20c5959b1007`（两者逐字节相等，A）。1373 ops，**1332 落 `@zhuget(330)`、41 落 `@hostt`**。
3. **`@hostt` 的 41 颗不构成能力缺口**（A，§7.4）：`SliceDynamic` 36＋`Input`／`Gather`／`RopeDynamic`／`CausalMask`／`Output` 各 1——全是 Qwen3 现役那颗也留在 host 的动态控制类；**1332 颗 HardOp 全上了 ZG**。
4. **为迁就 parser 付了三笔债**（A，§7.2／§7.3／§7.5）：① KV head 2→8 复制（ERNIE 原生 2 个 KV head，parser 只认得 Qwen3 那套几何）；② 补 4 条**没人消费**的 `past_key_values` 声明，把输入条数从 38 抬到 42；③ RoPE 配对只能在**数据层**置换 qkv 权重列，因为 parser 不读 `rotary_interleaved` 属性。
5. **PC 侧数值通**（A，§4／§8.2）：本机 iCraft Host 后端逐步链 7 步，**compiled 与 ORT 替身 7/7 argmax 逐位相同**（`ort_vs_compiled` cos 0.999995～0.999999）；除首步外对 HF fp32 的 cos 0.9876～0.9972。
6. **首步是坏的，根因不是精度**（A，§8.5）：首步 `comp_vs_hf` cos＝**−0.253818**，而替身对同一输入给同样坏的值 ⇒ 与编译无关，是**喂进去的第 0 槽是全零 pad**（厂商驱动约定）。
7. **校机实测：可用性判到了"能用但窄"**（A，§9.1～§9.3）：60 题、5 列对照。`native`（原始 genai 件）对 HF fp32 `cos` 0.999999／`top1_match` **60/60**；`e41_chain`（逐 token 链）`cos` 中位 0.816、`top1_match` **23/60**、`acc_top1` 0.0667。中文 76 token 文本困惑度 `ppl_chain` 36.297 vs `ppl_hf32` 33.758（argmax 一致 65/75）；**英文 71 token `ppl_chain` 153.215** vs `ppl_hf32` 29.455（一致 30/70）⇒ 退化主要落在英文侧。
8. **修首步＝slot1→slot0 复制，收益已量化**（A，§9.6）：替身面 60 题裸提示 `top1==HF` 23/60→**40/60**，chat template 面 11/60→**56/60**、`cos` 中位 0.7853→0.9956。**10-31 更正**：我先前那句"REPAIR HURTS on 18/60"是把"18 题 cos 变差"错写成整体变差；实际是净收益（见 §10 第 3 条）。
9. **这个修复在编译件上复核过**（A，§9.7，10-31 新跑）：`compiled_padfix` 8 步——step0 与不修链逐字节相等（控制组），8/8 每步 argmax 与替身完全一致，两种走法首 token 都取 23、与 HF 同；`TV` 0.03192→**0.02512** 变好，`cos` 0.992162→0.983253、`maxabs` 1.857→3.090 变差 ⇒ **一个提示上判不了"一定更好"，但机制在真件上成立**。
10. **门 4 通**（A，11-01，§14）：板上 `precheck=True`、`apply` 46.4 s；**逐 token 与 PC Host 后端对上**（padfix=0 七步、padfix=1 八步两模式都对上）；稳态 **0.42 s/步＝2.38 token/s**，64 token 长跑第 62 步吐 eos、`VmHWM` 不涨；padfix 板上 A/B 后**默认开**（关→时间戳循环，开→55 token 通顺到 eos）。**注意可复现的是板端**：两次板跑的 logits 极值逐位相同；**PC Host 后端复跑会对不上自己**（同一 42 路输入 |Δ| 均值 0.177／最大 0.668，非 padfix 那 7 步里有 3 步的 top-2 间距小于这个抖动）⇒ "全同"只在间距够宽的步上成立。**板上读数正本＝`wiki/PS/板端我方代码与产物台账.md` §廿五**，本页只记流程与合同。

## 1. 主机与环境（A，10-30 建线直读；10-31 分工沿用）

| 角色 | 机器 | 干什么 |
|---|---|---|
| 编译主机（唯一） | 本机 `C:\Icraft\CLI v3.39.0` | 门 1 改写链、五段编译（`icraft-optimize`／`quantize`／`adapt`／`generate`）、`icraft-run --backends Host` 前向 |
| 导出＋实测 | 校机 `ssh school-guest`（Windows，cmd 默认） | ModelScope 取权重、genai builder 导出、HF fp32 基准、60 题评测、长度／困惑度扫描 |
| 目标板 | `192.168.1.113` | **零读数**（门 4 未执行） |

- 本机**没下 GB 级件、不跑链路实测**（她的口径）；`model.onnx`(103,511 B) 与 `model.onnx.data`(755,105,792 B) 留在校机，编译根里那份是 §7 五段用的副本（体积件被 `.gitignore` 挡回，不入库）。
- venv＝复用 Qwen3 线那个：`D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe`；Python 3.12.15／transformers 5.19.0／torch 2.14.1+cpu／onnx 1.23.2／onnxruntime 1.31.0／onnxruntime-genai 0.17.1（产物 producer 记 `0.17.1+83de55a`）／modelscope 1.40.2／numpy 1.26.4。
- `ort.get_available_providers()` ＝ `['AzureExecutionProvider', 'CPUExecutionProvider']` ⇒ **该 venv 无 CUDA EP**，`-e cuda` 只是"哪种精度＋EP 组合"的标签，不代表校机有 GPU。读数来源 `scripts/ver_probe.py`（`pip list | findstr` 在那个 venv 取不到数，改用 `importlib.metadata`）。
- 校机 `D:` 空闲 1,645,066,383,360 B（约 1.5 TiB，A）。工作根 `D:\yzd\WCIS\MOD\ernie\`。

## 2. 权重快照 ＋ 导出（校机，A）

```
# 1) 下载（ModelScope 官方工具，allow_patterns 只收文本侧＋单文件权重）
D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe D:\yzd\WCIS\MOD\ernie\dl_ernie.py
#    -> 12 件 734,582,404 B；model.safetensors 721,514,672 B（dir 直读）
#       终端报 TOTAL_MB 700.552 / SECS 118.7 —— 该轮 stdout 未落盘，这两数按 B 用

# 2) 导出（scripts/export_a.ps1，日志落 export_a.log）
& 'D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe' -m onnxruntime_genai.models.builder `
  -i 'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT' `
  -o 'D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda' `
  -p fp16 -e cuda -c 'D:\yzd\WCIS\MOD\ernie\cache_dir'
```

- **不需要** Qwen3 那句 `--extra_options fuse_qk_norm_gqa=false`（日志 `Extra options: {}`，产物照样是融合 `GroupQueryAttention`＋`do_rotary=1`，A）。原因＝ERNIE 没有 per-head QK-Norm（§3 的算式核对）。
- **`-e cuda` 必带**，与 Qwen3 同：`-p fp16 -e cpu` 会退化成 `MultiHeadAttention`＋显式 `RotaryEmbedding`＋多一个 `position_ids` 输入（[[Qwen3-Instruct按Qwen3配方迁移]] 门 1，A）。这条对 ERNIE **没重跑过**（B，照搬结论）。
- 日志判据：**末尾 `EXIT=0`**（A）。日志第 4～9 行那段 `python.exe : … NativeCommandError` **不是失败**，是 PS5.1 把 stderr 包成错误记录（脚本里 `$ErrorActionPreference='Continue'`）。
- **编码坑**：`Tee-Object`（PS5.1）落 UTF-16LE，而 `Out-File -Append -Encoding utf8` 追加的 `EXIT=0` 是 ASCII ⇒ 同一文件两种编码。`logs/logs_export.txt` 是转好的 UTF-8/LF，进度行按"整行含 `|`"滤掉。

产物 6 件（校机 `dir`＋`certutil -hashfile MD5`，A）：

| 件 | 字节 | MD5 |
|---|---|---|
| `model.onnx` | 103,511 | `6f07fa164050bb4785584b90d0bb458e` |
| `model.onnx.data` | 755,105,792 | `788afdcb6bfd2502ceda667bcf0d1aac` |
| `tokenizer.json` | 11,185,815 | `3de03196f2973544dc3bbed4633586bd` |
| `tokenizer_config.json` | 20,770 | `913d614aa5a932509f8f6e03b6d8ff87` |
| `genai_config.json` | 1,693 | `6f9e65940dec312dd8b9ba383c588577` |
| `chat_template.jinja` | 777 | `3cde164445bb9f4af90f03c91b5d95db` |

`export/ernie_fp16_cuda/` 里另有 5 件（`config.json`／`generation_config.json`／`added_tokens.json`／`special_tokens_map.json`／`tokenizer.model`）是从权重快照目录补进来的，md5 与校机快照逐件相等（A）。

## 3. 图清点（`logs/logs_census.txt`，A）

`scripts/census_onnx.py` 同轮跑两颗：ERNIE 与校机 Qwen3 的 `qwen3_fp16_cuda`（对照用）。

| 字段 | ERNIE 本轮导出 | Qwen3 现役编译所用件 |
|---|---|---|
| ir／opset | ir10／(`ai.onnx` **22**, `com.microsoft` 1) | ir10／opset 21（10-12 原件；10-19 重导是 22） |
| producer | `onnxruntime-genai/0.17.1+83de55a` | 同 |
| 节点／初始化器 | **210／130** | 520／256 |
| 输入／输出 | **38／37** | 58／57 |
| 算子种类 | **13**（MatMul 91、SkipSLN 36、Mul 36、GQA 18、Sigmoid 18、Constant 2、Cast 2、Gather 2、ReduceSum/Sub/Shape/Transpose 各 1、SLN 1） | 15（多出 `Reshape` 112、`Split` 28） |
| GQA 属性 | `num_heads=16`、`kv_num_heads=2`、`scale=0.0883883461356163`（＝1/√128）、`local_window_size=-1`、`softcap=0.0`、`do_rotary=1`、**`rotary_interleaved=1`** | `kv_num_heads=8`、**`rotary_interleaved=0`**，其余同 |
| rope 表 | `cos_cache`／`sin_cache` 各 `fp16 (131072, 64)`＝16,777,216 B，**external 在 `.data`**（offset 509,739,008／526,516,224） | `(40960, 64)` 各 5,242,880 B |
| KV 第 4 轴 | **`dim_param: "kv_cache_dim"`**（36 进＋36 出全部如此） | `dim_value: 128`（10-12 原件烘死） |

- 算式核对（A）：`MatMul 91 ＝ 18×5＋1`（融合 qkv ⇒ 每层 qkv/o/gate/up/down 五颗 ＋ `lm_head` 一颗）；Qwen3 是 `7×28＋1＝197`，因为它的 q/k/v 分三颗。对照页 §7 那条"Matmul＝7×层数＋1"**对 ERNIE 不适用**，得按融合形态用 `5×层数＋1`。
- 算式核对（A）：norm 类 `1＋36＝37 ＝ 18×2＋1` ⇒ 没有 QK-Norm 这条在图里成立。
- 算式核对（A）：qkv 投影宽 `1024＋2560＝3584 ＝ 128×(16＋2＋2)` ⇒ Q 16 头、K/V 各 2 头、head_dim 128，与 GQA 属性互证。
- `.data` 对账（A，**10-31 按张量级读数更正**）：`model.safetensors` 头部 164 张量逐张求和＝**360,748,032 个元素**、全 BF16 ⇒ 权重字节 ＝ 360,748,032×2 ＝ **721,496,064**（文件 721,514,672 − 18,608 B 的头部与对齐，算式对平）。于是 `.data` 的未核余量 ＝ 755,105,792 − 721,496,064 − 33,554,432（两颗 rope 表）＝ **55,296 B**（＝27,648 个 fp16，成分仍未核）。原先这里写的"36,688 B 未核"是拿 safetensors 的**文件字节**当权重字节算的（36,688 ＋ 18,608 ＝ 55,296，两处差额对得上）；对照页 §6.3 那句"参数算式比 safetensors 少 19,632 B"是同一个错的另一面（18,608 ＋ 1,024；那 1,024 B ＝ `103,424×1,024` 少写 512 个元素，正确值 105,906,176 而非 105,905,664）。**参数量升到 A：360,748,032 ≈ 0.36 B（164 张量，全 BF16）。**

## 4. 门 0：对 HF 基准（校机导出件，`logs/logs_verify.txt`，A）

`scripts/verify_onnx_ernie.py` ＝ Qwen3 线 `compile/verify_onnx.py` 改三处（`SRC`／`ONNX_DIR`／`PROMPT`），方法同源：**golden 用 `use_cache=False` 在全序列上重算**，所以 DECODE 那一步测的是"ONNX 自己的 KV 缓存算术上连不连续"。

- 提示：`请用一句话介绍量子计算。` ⇒ **7 token** `[94515, 94036, 31282, 5502, 44284, 3458, 93977]`
- golden＝HF `Ernie4_5ForCausalLM` **fp32**；ONNX＝**CPUExecutionProvider** 上的 fp16（该 venv 无 CUDA EP，§1）。
- prefill 的 KV 喂**零长度** `(1, 2, 0, 128)` fp16 ×36；decode 把上一步的 `present.*` 原样喂回，`attention_mask` 长 8、`input_ids` 长 1。

| 步 | maxabs | relmax | cos | top5（ONNX vs HF） | top1 |
|---|---|---|---|---|---|
| PREFILL | 0.02105 | 0.00114 | **0.999999** | `[23, 44284, 2, 93919, 269]` **逐字相同** | 同（23） |
| DECODE ×1 | 1.56689 | 0.12047 | 0.991060 | `[23, 93942, 4, 3, 93993]` vs `[23, 93942, 93993, 1300, 4]`（前 2 同、第 3 位起顺序不同） | 同（23） |

判读（A 现象＋B 归因）：与 Qwen3 线同步骤同量级 ⇒ 归因 **fp16 累加误差随步数放大，不是图错**。**10-31 补充**：§9.3 的长度扫描（L=4～76）显示这个归因只在部分长度成立——`chain` 对 HF 的 cos 在 L=4 是 0.995798、L=8 掉到 0.997136 但 top1 变 rank 1、L=76 又回到 0.998803 ⇒ **误差与步数不是单调关系**，"随步数放大"这句降级为 B。

## 5. `interleaved`：我判错的那条，以及它只翻回来一半

- **源码面**（A，`scripts/probe_*.py` venv 内直读）：`models/builders/ernie.py:14` 无条件 `self.rope_attrs["interleaved"] = 1`；HF `models/ernie4_5/modeling_ernie4_5.py` 的 `rotate_half` 是 `x1=x[..., 0::2]`／`x2=x[..., 1::2]` ＋ `stack((-x2, x1), dim=-1).flatten(-2)`，**相邻配对＝interleaved**，旁边就是 `# Interleave them instead of usual shape`。Qwen3 那份同名函数才是半轴切分 ⇒ **同一个名字在两家 HF 里指两件不同的事**，我原来把 Qwen 的形状套到了 ERNIE 上。
- **消费面**（A）：`rope_attrs["interleaved"]` 在 builder `base.py:329` 默认 0、在 `:3439`／`:3493`／`:4203`／`:4446` 被读，最终成为 GQA 的 `rotary_interleaved`；`make_rotary_embedding_caches_from_scratch` 造表时 `emb=cat((freqs,freqs))` **不重排权重**（`:19-20` 的 `compression_ratio` 分支对这颗不触发）。⇒ 在 **ORT／genai 侧**烘死表与 interleaved 内核是同一套约定，不存在"表对、内核错"。
- 数值面（A）：PREFILL cos＝0.999999 且 top5 逐字相同（§4）。rope 约定若错一位，对不出这个数。
- **但这条结论只管到 iCraft 门前**（A，§7.3）：iCraft 的 parser **把这个属性整个丢掉**，于是"interleaved 的图"进了"按 split-half 执行的编译链"，**在 icraft 侧确实错了位**，得靠数据层置换修。**方法论保留**：这类错只在数值层暴露、结构门禁抓不到。

## 6. 门 1：改写链（13 步，全部 `compile/` 内现役脚本，A）

顺序不是编排出来的偏好，是**每一步只解掉当时那一条真实报错**；每步的"为什么是这一步"写在该脚本 docstring 里，这里只列读得出的因果。基座＝§2 那颗 `model.onnx`；产物逐份在编译根 `fmodel/e_vN/`。

| # | 脚本 | 改什么 | 逼出它的那条读数 |
|---|---|---|---|
| 1 | `meta_variant.py` | ir 10→8、`ai.onnx` opset 22→17、把 36 颗 `SimplifiedLayerNormalization` 移进 `com.microsoft` | Qwen3 线同款工序；不改则 `icraft-parse` 直接拒载 |
| 2 | `bake_kv_dim.py` | 符号轴 `kv_cache_dim` → 定长 128 | parse 头打印 ERNIE 是 `[1,2,0]` 三轴、Qwen3 是 `[1,8,0,128]` 四轴 |
| 3 | `split_qkv.py` | 融合 qkv 拆成独立 Q/K/V 输入 | parser 自己 lowering GQA 时的切片方式（parse 日志）只认拆开形 |
| 4 | `constantize_split_sizes.py` | Split 的 `sizes` 从 inline initializer 改成 Constant 节点 | 拆完的 Split 出现在 op_id 4..8 但后续段炸 |
| 5 | `shrink_rope.py` | rope 表声明几何收窄（保留原字节做前缀） | 拆 qkv 后不再 AV，改在降**第一颗 GQA** 时 `std::bad_alloc`；这颗唯一比 Qwen3 大的数就是 131072 行的 rope 表 |
| 6 | `patch_attr.py` | 诊断：把 `rotary_interleaved` 1→0（`.data` 硬链接） | 只为定位 AV 落在 rope 分支；**这条改动至今没回滚**，见 §7.3 |
| 7 | `tile_kv_heads.py` | K/V 头 2→8 复制 | AV 之后仍是 bad_alloc；逐个候选标量筛下来只剩 `kv_num_heads` |
| 8 | `reshape_gqa_inputs.py` | GQA 的 Q/K/V 前面垫恒等 `Reshape` | 图上下文清点（`cmp_graph_context.py`）与能编的图只差这一处 |
| 9 | `decouple_gqa_inputs.py` | 把一层 GQA 从生产者上摘下来单独喂 | 判"是上下文还是这颗本身"——控制实验，不是交付步骤 |
| 10 | `add_head_roundtrip.py` | 补上 parser 实际要求的 GQA 生产者拓扑 | 上一步摘出来照样编不过 ⇒ 要求在生产者侧 |
| 11 | `crop_layers.py` | 裁到前 N 层，保持命名与序号连续 | 全 parse 日志清点是一条干净二分：Qwen3 系（58 输入）过、ERNIE（38/40）不过 |
| 12 | `pad_cache_inputs.py` | 补 4 条**无人消费**的 cache 声明，38→42 | 裁层后仍挂 ⇒ 变量不是层数而是**声明条数**（§7.2） |
| 13 | `permute_rope_heads.py` | 每个 Q/K head 块的 head_dim 轴置换 `PERM=[0,2,…,127,1,3,…,127]` | parser 不读 `rotary_interleaved`（§7.3） |
| — | `mk_ort_variant.py` | 出一份"能改回元数据"的 ORT 替身 | `e_v35` 在 ORT-genai 拒载（`SimplifiedLayerNormalization(-1) is not registered`）⇒ 替身只为跑前对照 |

`e_v1`～`e_v41` 全部留档（编译根 `fmodel/` 68 个目录，其中 `e_v*` 47 个、编号 1～41 连续无缺；`*_parse.toml` 63 份，A）。二分控制件（`mk_minimal_gqa.py`／`mutate_qwen3_topo.py`／`roundtrip_control.py`／`drop_attention.py` 等）都留在 `compile/`，因为结论要能重放。

## 7. 门 2＋3：五段编译（本机，A）

### 7.1 到案的件

`E41-decoder`（`compile/E41_parse.toml` 902 B／md5 `00a4ce9f4205e095676c6154d3c7d58e`，五段各带自己的文件名与 `jr_path`，由 `mk_toml.py` 从基座 toml 文本改写；**10-31 起这份 toml 已入本仓 `compile/`**，之前只在编译根）：

- toml 里烘死的三个数（A，件直读）：`max_tokens_length = 512`＝**KV 窗口**（所以 §9.4 那条"改 `SLOTS`"必须重编才对图有效）、`max_prompt_length = 32`、`qdtype = "fp16"`＋`quantized = true`⇒ 门 2 的量化档只有 fp16 这一种走过；`llm_mode = true`／`target = "zhuge"`／`pass_off = "icraft.MergeTransDataOpSeqPass"`（adapt 段唯一关掉的 pass）。

| 段 | `.json` B | `.raw` B |
|---|---|---|
| parsed（decoder） | 795,009 | 1,009,070,088 |
| optimized | 661,133 | 1,009,088,484 |
| quantized | 665,851 | 1,009,088,484 |
| adapted | 1,443,443 | 1,009,088,484 |
| **_ZG** | **1,637,490** | **1,054,488,148** |
| prefill（只到 parsed） | 797,867 | 991,129,608 |

- `.raw` 从 optimized 起三段同尺寸（1,009,088,484），ZG 段多出 45,399,664 B（＝1,054,488,148 − 1,009,088,484，A）＝ZG 自己的参数布局。
- **`params_md5` 与 `.raw` 的 md5 逐字节相等**（`6c0fe5afb0d744a9dfdd20c5959b1007`，A：`zg_census.py` 报的 md5 对上本机 `hashlib.md5` 直读）⇒ 件自证一致。
- `ai_target @zhuget(330)`／`framework_version opset17`／`icraft_version v3.39.0`／`llm_mode True`／`swap_mode False`（A）。
- **五段全通的只有 E38 与 E41**；E40 停在 adapted（没出 ZG）；E15／E37 与全部 `Q3*` 对照件只到 parsed。**decoder-only 才走五段**，prefill 支路到 parsed 为止（厂商流程，A）。

### 7.2 债一：cache 输入按**声明条数**注册（不是按层数）

- 现象（A）：同一张图，输入声明 ≤40 条时 parser 只注册 2 个 network input（`input_ids`＋`token_dynamic_fm`），然后第一个 GQA 访问违例、且**只落 prefill 日志、没有 ftmp_match**；≥42 条时全部接上。
- 与层数无关这条有对照：28 层原图 58 条过、裁层后不过、补 4 条假声明（38→42）过（`crop_layers.py`／`pad_cache_inputs.py` 的 docstring 记了这条二分）。
- 代价（A）：4 条 `dead` 输入 v_id 38～41 **留在输入清单里**，每步多喂 4×1,048,576 B ＝ **4 MiB 死数据**；板端驱动必须喂满 42 路，因为运行时会先检查 `network_inputs_size == tensor_inputs_size` 再派发。
- 尚未排除的（C）：全 bin 搜 `past_key_values` 字面量零命中 ⇒ 阈值的可能来源没查到，"40/42"是实测边界，不是解释。

### 7.3 债二：parser **不读** ONNX 属性，RoPE 配对只能改数据

- 控制组（A）：两份 IR 只差网络名（2 字节）、`rotary_interleaved` 分别 1／0 ⇒ **logits 逐比特相同** ⇒ 属性被整个丢掉，改属性是无效操作。
- 四列对照（A，`cmp_host_hf.py`，同一提示、同一喂法）：

| 列 | maxabs | cos |
|---|---|---|
| native genai 件／`ri=1` | 1.459 | 0.99411 |
| `ri=0` 改写件（E38，编译链按 split-half 执行） | ≈4.2 | ≈0.96465 |
| **置换后**（E41，qkv 权重列按 `PERM` 重排）干净缓存 | **0.0206** | **1.00000** |

- ⇒ 结论：`ri` 的语义在 icraft 侧必须靠**数据置换**兑现；§5 那条"interleaved 本来就对"**只在 ORT／genai 侧成立**。第 6 步那次 `ri 1→0` 的诊断改动**至今没回滚**，实际生效的是第 13 步的权重置换。

### 7.4 债三：算子归属读数

- `@zhuget(330)` 1332 颗，全是 `HardOp`（融合核）；`@hostt` 41 颗＝`SliceDynamic`×36 ＋ `Input`／`Gather`／`RopeDynamic`／`CausalMask`／`Output` 各 1（A，`zg_census.py` 直读）。
- 判读：41 颗 host 全是 KV 切片的动态控制类，与 Qwen3 现役那颗同形 ⇒ **不是这颗模型独有的能力缺口**（B：对 Qwen3 那颗的同类清点在对照页）。
- `(16Q／2KV→平铺 8KV／head_dim 128)` 这个组合现在**有读数了**（旧 §7 那条"零读数"作废）。

### 7.5 KV 几何被抬了一倍

- 平铺 2→8 ⇒ 每颗 cache `@fp(16)[1,8,512,128]`＝1,048,576 B（A）。对账：若按 ERNIE 原生 2 KV head 只要 `[1,2,512,128]`＝262,144 B，8 头是它的 4 倍（算式，B）。⇒ **上板时 CMA／带宽预算按 8 头算，不按 2 头算**（门 4 未验）。

## 8. PC 侧：`feed` 合同 ＋ Host 后端逐级链

### 8.1 `feed` 合同（读自 `E41-decoder_adapted.json` ＋ `gen_e41_chain/step*/feed/`，A）

- 42 条网络输入，形态计数：`(1,1)`×1 ＋ `(1,3)`×1 ＋ `(1,8,512,128)`×40；dtype `@fp(32)`×2 ＋ `@fp(16)`×40。
- 每步喂入 **41,943,056 B** ＝ 4 ＋ 12 ＋ 40×1,048,576（A：`step0/step1/step7/feed` 三处逐目录实测，42 个文件、尺寸集 `{4, 12, 1048576}`，三步全等）。
- `v_id 0` ＝ 本步 token（fp32，4 B）；`v_id 1` ＝ `token_dynamic_fm`＝`[新 token 数, 已填槽＋1, 0]`；**元素 1 同时是写槽与绝对位置**，元素 2 没人读。
- 36 条真 cache ＋ 4 条 dead（v_id 38～41，`never consumed`，仍必须喂）。
- 输出 logits `@fp(16)[1,1,103424]` ＋ 36 条 `present`。
- **ZG json 把名字抹平了 ⇒ 驱动只能认序号，不能认名字**（A）。
- 提示 token 索引 t → 写槽 t＋1，**槽 0 永久全零**（厂商驱动约定，A）。
- `--pos-offset 1` 不可改 0：元素 1＝0 会让动态切片炸成 512× 并在 op_id 4525（Reshape）失败（Qwen3 线同形读数，A；ERNIE 未单独重跑，B）。

### 8.2 逐级链（`compile/host_gen.py`，A）

正本＝本仓 `compile/host_gen.py`（基座 `S:\work\WCIS\workenv\qwen3\host_gen.py` 304 行／`9ec6a0c9…`，那份真跑过 E41；`repos/Qwen3-0.6B-Instruct/compile/host_gen.py` 274 行／`c9a86967…` 早于 dead-input 处理，**跑不了带 pad 的图**）。

不修（`--padfix` 关）7 步，提示 `[94515,94036,31282,5502,44284,3458,93977]`：

| step | fed | past_len | argmax | 秒 |
|---|---|---|---|---|
| 0 | 94515 | 1 | 4 | 11.8 |
| 1 | 94036 | 2 | 18196 | 11.6 |
| 2 | 31282 | 3 | 31123 | 11.2 |
| 3 | 5502 | 4 | 94042 | 11.5 |
| 4 | 44284 | 5 | 94113 | 11.5 |
| 5 | 3458 | 6 | 12022 | 11.5 |
| 6 | 93977 | 7 | **23** | 11.8 |

- 平均 11.557 s/步（算式：(11.8＋11.6＋11.2＋11.5＋11.5＋11.5＋11.8)/7 ＝ 80.9/7，A）＝ 0.0865 token/s（纯前向，未计分词与解码）。
- `icraft-run` 侧：`--backends Host`、`--dump_output --dump_format SFB --dump_ftmp <v_ids>`（收窄到 logits＋36 present），**SFB dump 一律加宽成 fp32**（Qwen3 线实测：`@fp(16)` 的 `[1,8,512,128]` 倒出 2,097,152 B）⇒ `read_dump()` 按"文件字节／元素数"反推 dtype，再存回 feed 声明的 fp16。

### 8.3 关键控制：compiled ＝ ORT 替身，7/7（A，`check_e41.json`，`eval_ernie.py --mode check`）

| step | fed | `ort_vs_compiled` cos | maxabs | argmax compiled／ort／HF |
|---|---|---|---|---|
| 0 | 94515 | 0.999999 | 0.0352 | 4／4／**23** |
| 1 | 94036 | 0.999996 | 0.0420 | 18196／18196／18196 |
| 2 | 31282 | 0.999997 | 0.0430 | 31123／31123／31123 |
| 3 | 5502 | 0.999998 | 0.0391 | 94042／94042／94042 |
| 4 | 44284 | 0.999999 | 0.0352 | 94113／94113／33576 |
| 5 | 3458 | 0.999997 | 0.0547 | 12022／12022／93977 |
| 6 | 93977 | 0.999995 | 0.0449 | 23／23／23 |

⇒ **替身可预测编译件**（argmax 全同、cos ≥0.999995），所以 §9 那一大批"替身面"评测有意义；代价是它比编译件快 **117×～204×**（11.557/0.0984 与 11.557/0.0568，A）。

### 8.4 首步坏的，不是精度坏的（A）

同一条链的首步 `comp_vs_hf` cos ＝ **−0.253818**／maxabs 15.5562，而第 1～6 步 cos 0.9876～0.9972 ⇒ 断点精确落在"槽 0 是全零 pad"这一步。`mk_padrand.py`（把 pad 槽填随机数）＋ `cachemap.json`（5 例 ×18 层 `slot0_maxabs` 全 0.0）⇒ **图真的会去attend 那格全零**，不是被 mask 掉。第 6 步之后 `argmax` 与 HF 有两处不同（step4 94113 vs 33576、step5 12022 vs 93977），归因为链上累积（B）。

## 9. 校机实测（`eval/`，60 题＝`items.json`：成语／算术／常识／翻译／代码 5 类各 12，A）

评测方式：`golden` ＝ HF fp32 全序列 `use_cache=False`；五列＝`hf32`／`hf16`／`native`（原始 genai 件）／`e41_seeded`（HF 缓存灌进 e41 的平铺＋置换基座，喂末 token）／`e41_chain`（e41 逐 token 走，全零缓存起步）。`cos`／`maxabs`／`kl`／`tv` 都按 logits 原始量与 softmax 概率两套算。

### 9.1 裸提示（`eval_items.json`，A）

| 列 | `cos` 中位 | `cos_min` | `maxabs` | `kl` | `top1_match` vs HF32 | `acc_top1` | `mean_target_rank` |
|---|---|---|---|---|---|---|---|
| `hf16` | 0.999999 | 0.999996 | 0.0251 | 1e-05 | **60/60** | 0.2167 | 163.27 |
| `native` | 0.999999 | 0.999998 | 0.0198 | 1e-05 | **60/60** | 0.2167 | 162.27 |
| `e41_seeded` | 0.98184 | 0.874814 | 2.6713 | 0.22138 | 41/60（0.6833） | 0.2167 | 237.08 |
| `e41_chain` | 0.816042 | 0.312998 | 8.0894 | 1.22396 | **23/60**（0.3833） | 0.0667 | 1349.32 |

- 算式核对：`0.3833×60 = 23.0`、`0.6833×60 = 41.0`（A）。
- `acc_top1` 对 `hf32`／`hf16`／`native`／`e41_seeded` **四列同为 0.2167** ⇒ 0.2167 是**这颗模型本身在 60 题答案键上的上限**，移植没有动它；`e41_chain` 掉到 0.0667 是逐 token 链的额外损失。

### 9.2 chat template 面（`eval_items_chat.json`，A）

同一 60 题走 `chat_template.jinja`（`<|begin_of_sentence|>User: … Assistant: `）⇒ 提示长度中位 12.65 token。

| 列 | `cos` 中位 | `maxabs` | `top1_match` vs HF32 | `acc_top1` |
|---|---|---|---|---|
| `native` | 0.999999 | 0.0259 | 60/60 | 0.0 |
| `e41_seeded` | 0.994946 | 2.6031 | 58/60（0.9667） | 0.0 |
| `e41_chain` | 0.766122 | 13.9126 | 11/60（0.1833） | 0.0 |

- **两处公开更正**：① `acc_top1` 这列在 chat 面**对 `hf32` 自己也是 0.0** ⇒ 答案键是按裸提示建的，chat 面不能用 `acc_top1` 判好坏，只能看 `top1_match_vs_hf32`（这条错我在 §9.2 首次写的时候挂成了"模型不行"）。② 我先前把 `-PT` 读成 pretrain 底座——错，见 §10 第 1 条。
- `e41_seeded` 在 chat 面 58/60 与裸提示面 41/60 的差 ⇒ **提示越长，"灌 HF 缓存"这条路越接近可用**；坏的是从零开始的链。

### 9.3 长度扫描 ＋ 困惑度（`sweep.json`，A）

- 长度 L＝4／8／16／24／32／48／64／76：`chain` 对 HF 的 cos 0.995798／0.997136／0.997935／0.997660／0.998400／0.998477／0.995432／0.998803，`top1_match` 除 L=8（rank 1）外全 True ⇒ **长提示单步喂进去并不坏**。
- 与 §9.1 的 `chain` 差在哪：这里的"chain"也是逐 token，但喂的是 L 长的完整前缀，槽 0 毒化只污染 1/L 的注意力质量；60 题的平均提示长 4.87 token ⇒ **短提示被这一格拖死**（B，归因与 §9.6 的修复收益一致）。
- 困惑度（75／70 个计分位）：中文 `ppl_hf32` 33.758／`ppl_hf16` 33.762／`ppl_native` 33.737／`ppl_chain` **36.297**，argmax 一致 65/75；英文 29.455／29.451／29.451／**153.215**，一致 30/70 ⇒ **退化主要落在英文侧**（A 现象，归因未做，B）。`chain_ms_per_step` 56.8（中）／98.4（英）＝替身在 24 逻辑核上的速度。

### 9.4 缩短 cache 能不能救（`pad.json`，14 例，A）

同一 14 例在 `SLOTS` ＝ 512／256／128／64／16 五档下 **`cos`／`maxabs`／`rank_of_hf_top1`／`top1` 逐位相同**（例：item1 五档全 0.683071／15.2275／3650／94786）⇒ **缩短缓存不改变任何读数** ⇒ 病根不在 512 槽的几何，改 `SLOTS` 这条路死；**不需要重编**（A，这条同时否证了"pad 槽离写入位置太远"的猜测）。

### 9.5 基座与误差分解（`cachemap.json`／`decomp.json`，A）

- 灌缓存的基座对照（5 例 ×18 层＝90 行）：**K 在置换基座** `cos` 中位 **0.97913**（min 0.38958），在自然基座 0.01783；**V 在自然基座** 0.98851，在置换基座 0.00404 ⇒ **置换只该作用在 Q/K，V 不动**（`eval_ernie.py:seed_cache` 正是只 `kh[..., PERM]`，A）。K 的 0.979 而不是 1.0 ＝ fp16＋GQA 内核的残差（B）。
- 误差分解（14 例 ×`chain`／`seeded` 两列）：`chain` 的 `maxabs_raw` 中位 12.019，减掉常数 `shift` 后 11.5605 ⇒ **常数平移只解释一小部分**（log-softmax 下平移本身免费）；`pearson` 中位 0.66109／`tv` 0.46815／`kl` 0.78904。同表 `seeded` 列 `maxabs_raw` 1.523／`pearson` 0.99096／`tv` 0.06287 ⇒ **链的误差是形状错，不是刻度错**（B）。

### 9.6 padfix：替身面 60＋60 题（`padfix.json`／`padfix_chat.json`，A）

修复动作（纯驱动约定，**不重编**）：token 0 连喂两次；每次 forward 之后把**槽 1 复制进槽 0**。窗口＝{self, self} ⇒ 位置 0 只跟自己算注意力。残差＝token 0 在后续整条链里带**双份权重**。步号映射：step0 中毒步（丢弃）、step1 重喂（它的 argmax 就是位置 0 的 logits）、step k≥2 喂 `prompt[k−1]`、元素 1＝k ⇒ 走满 `len(prompt)+1` 槽。

| 面 | 题数 | `cos` 变好 | `tv` 变好 | `top1==HF` 不修 | `top1==HF` 修 | argmax 不变 | `cos` 中位 不修→修 | `tv` 中位 不修→修 |
|---|---|---|---|---|---|---|---|---|
| 裸提示 | 60 | 43（71.7%） | 42（70.0%） | 23（38.3%） | **40（66.7%）** | 21 | 0.9781→**0.9910** | 0.2680→**0.1623** |
| chat | 60 | **60（100%）** | **60（100%）** | 11（18.3%） | **56（93.3%）** | 12 | 0.7853→**0.9956** | 0.8010→**0.0839** |

- 贪心 24 token 长链（每面 3 条样本）：对 HF 的平均一致位 裸 0.125→**0.389**、chat 0.083→**0.292**；更好 2／更差 0／相等 1（两面同）。
- 算式核对：`40/60＝0.667`、`56/60＝0.933`、`43/60＝0.7167`、`60/60＝1.0`（A）。

### 9.7 padfix 在**编译件**上的复核（10-31，`eval/compiled_padfix/`＋`eval_padfix_compiled.py`，A）

跑法：本机 `compile/host_gen.py --padfix`（新增开关：`chain_past(..., padfix=True)` 里 `arr[:, :, 0, :] = arr[:, :, 1, :]`；walk 前置一个重复 token；`past_len` 对 i≥1 减 1），把 8 份 `logits.npy` 推校机打分。

| step | fed | past_len | argmax | | HF 同位 |
|---|---|---|---|---|---|
| 0 | 94515 | 1 | 4 | 中毒步，丢弃 | 23 |
| 1 | 94515 | 1 | **23** | 重喂＝位置 0 | 23 |
| 2 | 94036 | 2 | 8231 | | 18196 |
| 3 | 31282 | 3 | 31123 | | 31123 |
| 4 | 5502 | 4 | 2640 | | 94042 |
| 5 | 44284 | 5 | 33576 | | 94113 |
| 6 | 3458 | 6 | 12022 | | 12022 |
| 7 | 93977 | 7 | **23** | 首 token | 23 |

三条控制（全部通过，A）：① `step0` 与不修链**逐字节相等**（`np.array_equal` True）⇒ 驱动只改了修复本身；② 不修链 7/7、修复链 8/8 每步 argmax 与替身**完全一致**；③ 替身预测的首 token 位移 23→23 ＝ 编译件实测 23→23。

对 HF fp32 打分（末位＝位置 6 的 logits）：

| 列 | cos | maxabs | TV | top1==HF | rank |
|---|---|---|---|---|---|
| compiled plain | 0.992162 | 1.8572 | 0.03192 | True | 0 |
| compiled padfix | 0.983253 | 3.0902 | **0.02512** | True | 0 |
| twin plain | 0.992094 | 1.8689 | 0.03172 | True | 0 |
| twin padfix | 0.983194 | 3.0824 | 0.02535 | True | 0 |
| padfix vs plain | 0.972357 | 4.1484 | 0.02532 | True | 0 |

- 判读（A 现象＋B 归因）：**机制在真件上成立**（step0 由 4 变 23、与 HF 同；两套实现的 argmax 链 8/8 相同；编译件与替身的读数差 `cos` ≤6.8e-05），但**这一个提示上 `cos`／`maxabs` 反而变差**（TV 变好），首 token 两者都是 23 ⇒ **单提示不足以判"一定更好"**；收益判据取 §9.6 的 120 题面。
- 计时：修复链 8 步 13.4／13.3／13.5／12.7／13.1／13.2／13.0 s（step0 冷启 31.1 s，不计入）比不修链的 11.2～11.8 s 慢 **约 12%**（(13.4+13.3+13.5+12.7+13.1+13.2+13.0)/7＝13.17 vs 11.557，A）＝多一步前向＋每步 40 次槽复制的代价；板端代价**未测**（C）。
- **驱动修复的独立证据**：`--padfix` 首版在 walk 走完后越界（`IndexError`，因为只索引了 `walk`），修成索引 `feed`（walk＋已生成 token）后，2 token 提示 ×`--max-new 2` 连跑 5 步到 step4 仍在生成（`eval/compiled_padfix/smoke_2token/`，A）。

## 10. 更正记录（我判错、现已否证的四条）

1. **`-PT` ＝ PyTorch 权重格式，不是 pretrain 底座**（A，两张 ModelScope 卡片并读）。原话 `"-Paddle" models use PaddlePaddle weights, while "-PT" models use Transformer-style PyTorch weights`；裸底座叫 `-Base-PT`（`Training Stage | Pretraining`），本颗是 `a text dense Post-trained model`／`Posttraining`。`baidu` org 37 个模型里 0.3B 只有 `Base-PT`／`Base-Paddle`／`PT`／`Paddle` 四档 ⇒ **不存在 Instruct／Chat 档，"换 Instruct 重导"没有对象**。
2. **`interleaved` 那条坑先被作废、又翻回一半**（A，§5／§7.3）：ORT／genai 侧约定本来就对；icraft 侧因为 parser 丢属性而确实错位，需数据层置换。
3. **"REPAIR HURTS on 18/60" 这句是错的**（A，§9.6）：真实读数是裸提示 60 题里 **43 题 cos 变好、17 题变差**，`top1==HF` 23→40；chat 面 **60/60 全变好**、`top1==HF` 11→56。我把一个"部分题变差"的计数写成了"整体变差"（那个 18 现在追不到出处，不解释，直接作废）。**结论方向反了**，以 §9.6 表为准。

4. **参数量算式与两处"未核余量"错在同一个地方**（A，10-31，§3）：先前拿 `model.safetensors` 的**文件字节** 721,514,672 当权重字节去对 `.data`，于是得到"多 36,688 B"；对照页 §6.3 的算式又把 embedding 写成 103,424×1,024＝105,905,664（少 512 个元素）。现在按件直读：safetensors 头部 **164 张量、全 BF16、合计 360,748,032 个元素** ⇒ 权重 721,496,064 B，`.data` 未核余量 **55,296 B**（＝36,688 ＋ 18,608，那 18,608 B 是 safetensors 自己的头部与对齐）。参数量从"B 级算式"升成 **A 级件读数**；`.raw` 的那句外推（≈970 MB，C）同时作废——实测 1,054,488,148 B ≈ 1,005.6 MiB（§7.1）。
## 11. 复跑

```
# 校机：导出（§2）＋清点＋门 0 基准
ssh school-guest "D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe D:\yzd\WCIS\MOD\ernie\census_onnx.py D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda"
ssh school-guest "D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe D:\yzd\WCIS\MOD\ernie\verify_onnx_ernie.py D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda"
# 校机：60 题五列／chat 面／padfix 替身面／编译件打分（件在 D:\yzd\WCIS\MOD\ernie\eval\）
D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe -X utf8 eval_ernie.py --mode check
D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe -X utf8 eval_padfix_compiled.py   # -> padfix_compiled.log

# 本机：五段编译（E41 现役 toml）＋逐级链
"C:/Icraft/CLI v3.39.0/bin/icraft-run.exe" --json <adapted.json> --raw <adapted.raw> --input <inputs.txt> \
  --backends Host --dump_output --dump_format SFB --dump_ftmp <v_ids> --log_path <dir> --log_time
python -X utf8 compile/host_gen.py --json S:/apps/llm/agentwork/ernie/compile/jr_path/E41/E41-decoder_adapted.json \
  --raw  S:/apps/llm/agentwork/ernie/compile/jr_path/E41/E41-decoder_adapted.raw \
  --work S:/apps/llm/agentwork/ernie/compile/gen_e41_chain --prompt "94515,94036,31282,5502,44284,3458,93977" \
  --max-new 1 --stop 151645,151643        # 加 --padfix 走修复链
python -X utf8 compile/zg_census.py S:/apps/llm/agentwork/ernie/compile/jr_path/E41/E41-decoder_ZG.json

# 板（192.168.1.113；动手前先看 113.txt 是不是 off）
scp board/e41_probe.py board/e41_infer_board.py root@192.168.1.113:/mnt/data/ernie/
ssh root@192.168.1.113 "E41STEP=3 timeout 600 python3 /mnt/data/ernie/e41_probe.py"          # 分步探针
ssh root@192.168.1.113 "cd /mnt/data/ernie && E41_PADFIX=1 E41_MAX_NEW=8 \
  E41_PROMPT=94515,94036,31282,5502,44284,3458,93977 python3 e41_infer_board.py"           # 点火
```

`compile/` 与 `scripts/` ＝现役正本；工作副本与逐里程碑日志（`logs/*.log` 的 parse/optimize/quantize/adapt/generate、UTF-16LE 原件 `export_a_raw.log`、`padfix_compiled.out`）在 `S:\work\WCIS\workenv\ernie\`（不入库）。

## 12. 还没做（一条都没执行，别当进度）

1. **板上直接吃中文（分词在板上做）**：板上 `tokenizers` 是 0.13.3，读不了 ERNIE 的 `tokenizer.json`（`data did not match any variant of untagged enum ModelWrapper`——新版 BPE 才有 `ignore_merges` 字段）。现役驱动是**喂 id、出 id**，分词解码在本机做（本机 `tokenizers` 0.23.3 正常）。要板上端到端跑中文，得升板上的 `tokenizers` 或换 sentencepiece 路线——**未执行**。
2. **量化档**：只出 fp16；int8／mix 一个都没试。
3. **长链逐级比对**：Qwen3 线那套 25 token `ref_hf.py --simulate --teacher`／`cmp_chain.py` 对 ERNIE **没跑**；§9.6 只有 3 条贪心样本 ×24 位。
4. **英文退化归因**：§9.3 的 `ppl_chain` 153.215 vs 29.455 只记了现象，没查是 tokenizer／rope 表长度（131072 行 vs 实际用到）还是平铺 KV 带来的。
5. **60 题的编译件面**：§9.6 的替身面 120 题、§9.7 的编译件面只有 1 个提示；60 题在编译件／板上都没跑（板上只跑了 3 条提示）。
6. **`e_v*` 中间件的入库**：编译根 68 个 `fmodel/` 目录**没进本仓**（体积＋可再生），只留 `E41` 的合同读数与本表；要复算门 1 的某一步得回编译根。

> **11-01 从这张表划掉的三条**（都已执行，读数在 §14／台账 §廿五）：门 4 板端全套；板端驱动（`board/e41_infer_board.py` 已写并上板）；padfix 是否设默认（板上 A/B 后定为默认开）。

## 13. 路径对照

| 内容 | 校机（正本） | 本机 |
|---|---|---|
| 权重快照 | `D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT\` | 不回传（ModelScope 官方工具可重下） |
| 导出件（含 `.onnx`/`.data`） | `D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda\` | `export/ernie_fp16_cuda/`（只小件；体积件 gitignore＋未回传） |
| 导出原件日志（UTF-16LE） | `D:\yzd\WCIS\MOD\ernie\export_a.log` | `S:\work\WCIS\workenv\ernie\export_a_raw.log`；转码版＝`logs/logs_export.txt` |
| 门 1 改写链／门 2～3 五段件 | — | 编译根 `S:\apps\llm\agentwork\ernie\compile\`（`fmodel/e_v*`、`jr_path/E41/*_ZG.*`，不入库）；脚本正本＝本仓 `compile/` |
| 逐级链（不修／修复） | `D:\yzd\WCIS\MOD\ernie\eval\compiled_chain`／`compiled_padfix` | 本仓 `eval/compiled_chain/`／`eval/compiled_padfix/`（`logits.npy` fp32，103,424 元素＋128 B 头＝413,824 B／件） |
| 评测读数 json | `D:\yzd\WCIS\MOD\ernie\eval\*.json` | 本仓 `eval/*.json`（逐件 md5 见 `MANIFEST.md5`） |
| 板端件 | — | 板上 `/mnt/data/ernie/jr_path/decode/E41-decoder_ZG.{json,raw}`（md5 与本机全等）／`/mnt/data/ernie/config/ernie/` 四件／`/mnt/data/ernie/e41_{probe,infer_board}.py`；本机源＝`board/`。台账 #218～#222 |

## 14. 门 4：板上（2026-11-01，A）

**读数不在这里**——板上件、逐项内存读数、逐 token 对齐表、A/B 对照都在 `wiki/PS/板端我方代码与产物台账.md` §廿五。本节只记**在 PC 上复算不出来、只有上板才会碰到的东西**。

**板端合同与 `host_gen.py`（§8）的三处差异**（都是实测，不是推断）：

1. **输出不是"逐步压实"，是整块回喂**。`host_gen.py` 把 `present` 整个张量 chain 回 `past`（槽位 `past_len` 写入，前 `past_len-1` 槽原样带过）；Qwen3 现役驱动 `q3_infer_board.py` 走的是"每步把第 i 槽抠出来塞回第 i 槽"的压实法。**这颗必须用整块法**——照抄 Qwen3 驱动会错位。
2. **`token_dynamic_fm` 三元组**：`[new_token_count, past_len, 0]`，`past_len` 是真的"已填槽数"（含 padfix 的重复首 token），`--pos-offset 1` 等价物是 `past_len = 1 + 已走步数`。喂 `past_len=0` 会让 `SliceDynamic` 炸到 512 倍并死在 `Reshape`。
3. **42 路必须喂满**：`v_id 38..41` 四条 `past_key_values` 声明在图里没人消费，但运行时先比 `network_inputs_size == tensor_inputs_size` 再派发 ⇒ **四条零张量也得占位**，每步多 4 MiB。

**驱动的两个实现要点**：

- 输入缓冲按 `net.ops[0].outputs[i].tensorType()` 一次性建好（40.0 MiB＝41,943,056 B），`present` 用 `item.read(buf, item.offset(), chunk.byte_size)` 直读回配对槽位，不新建 numpy 数组。前两路用 `xir.Layout("*C")`，其余用 `tensorType()`——与 `q3_infer_board.py` 同法。
- `E41_PADFIX`（**默认 1**，设 0 关）在**回喂时**做 `buf[:, :, 0, :] = buf[:, :, 1, :]`，并把首 token 重喂一次（`walk=[prompt[0]]+prompt`）；这与 `host_gen.py --padfix` 是同一约定，板上读数逐 token 相同（台账 §25.2）。默认值是在板上 A/B 之后（11-01）改的，代码里就是 1，不靠启动命令带环境变量。

**位流门禁这次怎么过的**：`/mnt/boot` 未挂载，`md5sum /mnt/boot/BOOT.bin` 这条路不可用；改判证据＝同日另一进程在同一块板打出 `Device Name: zg330aiu`，加本次 `openDevice` 秒回、`precheck=True`（台账 §25.1）。**下次上板仍应先查 `/mnt/boot` 是否挂上**，别把这套替代证据当常规通道。

**比对判据的边界（11-01 补，A）**：板跑两次 → 11 步 logits 极值逐位相同；**PC Host 后端同一 42 路输入两次跑 → |Δ| 均值 0.177、最大 0.668**（`p2_pc` vs `p2_pc_full`，步 0）。所以：

- "板端 ≡ PC"只能按 argmax＋top-2 间距讲，不能按"逐位等价"讲；间距 < ~0.7 的步（提示 1 的 7 步里有 3 步：0.2031／0.0469／0.2656）不作硬判据。
- 想用 PC 当金标准，至少要跑两次、只采信两次都一致的步——单次 PC 结果不能当"正确答案"。
- 板端自己才是可复现的那一侧，这是把 `host_gen.py` 当"参考实现"时最容易忽略的一点。
