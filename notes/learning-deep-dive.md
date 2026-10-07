# ReKV 深入学习文档

> 上游：`Becomebright/ReKV`（Shangzhe Di 等）
> 本 fork：`jasonchen505/ReKV`（2026-10-07 创建，默认分支 `main`）
> 论文："Streaming Video Question-Answering with In-context Video KV-Cache Retrieval"，ICLR 2025
> 学习日期：2026-10-07｜状态：只读学习，main 未动，本文件位于 `notes/learning` 分支
>
> 注：本文基于代码走读（上游 HEAD `1fd9a3d`）+ README；论文观点为转述。

---

## 1. 一句话定位

ReKV 是一个**training-free** 的流式视频问答（StreamingVQA）方案：插到现有 Video-LLM 上，让模型可以一边看视频一边编码、收到问题立刻回答，而不用等整个视频处理完、也不用为每个问题重复算一遍。核心 trick 是**把视频的 KV-cache 当作可检索的外部记忆**：存到内存/硬盘，问答时只把问题相关的 block 搬回 GPU。

## 2. 问题背景：StreamingVQA vs OfflineVQA

传统 VideoQA 是 offline 的：整个视频先一次性过模型，拿到问题再一起推理。长视频（MLVU 里有 ~9 小时）有两个要命问题：

1. **延迟**：必须看完才能答；
2. **重复计算**：每个新问题都要重新处理一遍视频（或至少重新做一次完整前向）。

ReKV 的目标是把"视频分析"和"问答"解耦：视频流进来就增量编码成 KV-cache 存好；问题来时，做一次**检索增强的**问答。两者甚至可以跑在不同进程/GPU 上。

## 3. 三板斧

### 3.1 Sliding-window attention（编码期降本）

流式编码视频时，每帧只 attend 有限个前帧：保留开头的 `n_init` 个 token 做 attention sink（StreamingLLM 式），加上最近的 `n_local` 个 token。评测默认 `n_local=15000`、`sample_fps=0.5`。这样编码是 O(窗口) 而不是 O(全长)，长视频也能流式吃进去。

### 3.2 KV-cache offload（防信息丢失）

滑窗只解决了"算得动"，没解决"记得住"——滑出窗口的帧信息就丢了。ReKV 把处理过的视频 KV-cache **全量存到 RAM/disk**：

- `MemoryUnit`：单个 KV block 的管理单元，CPU 常驻（可选 `pin_memory` 加速传输），按需搬到 GPU；
- `CudaCache`：GPU 上固定大小的块池，`alloc`/`delete` 管理，`idle_set` 跟踪空闲块；
- 搬运走独立的 `GLOBAL_STREAM` CUDA 流、带 event 同步，与计算 overlap；
- block 用 LRU 淘汰（`_remove_lru_blocks`），`max_cached_block` 控制上限。

代价是系统要求：`vm.max_map_count` 要调到 262144（mmap 大量文件），README 里明确写了这条。

### 3.3 In-context KV-cache retrieval（问答期提效）

这是论文标题里的核心创新。问答时**不**把全量 KV 搬回 GPU，而是只检索问题相关的 block：

- **Internal retrieval**（默认）：只把 question 文本喂给模型，用 question 的 query 向量（按 head mean-pooled）与每个 block 的代表 key 算余弦相似度，取 top-k。注意这里用的是**模型自己的参数**做检索器，不引入外部模型。
- **External retrieval**：调用方直接给 `retrieved_indices`（block 下标），跳过内部检索——给"有更好的外部检索器"的场景留的口子。
- **Block 与 chunk**：block 按帧切分（`block_size = n_frame_tokens = 196`，即一帧的 token 数）；`chunk_size` 把相邻 block 打包成组再做 top-k，保证时序连续性（默认评测 `topk=64, chunk_size=1`，可调）。

检索到的 block + init sink 拼成新的 KV 上下文，问题和生成都基于它做——这就是所谓的 "in-context" retrieval：被检索的 KV 直接成为解码的上下文，而不是先转成文本再喂。

## 4. 实现走读

```
model/
├── patch.py                 # monkey-patch HF attention，换上 rekv_attention_forward + RotaryEmbeddingESM
├── attention/
│   ├── rekv_attention.py    # 自定义 attention forward：编码走 ContextManager.append，问答走滑窗+检索
│   ├── kv_cache_manager.py  # CudaCache / MemoryUnit / VectorTensor / ContextManager（704 行，核心）
│   ├── rope.py              # RotaryEmbeddingESM
│   └── dot_production_attention/  # 多阶段点积 attention（triton/torch 双实现）
├── abstract_rekv.py         # 基类：encode_init_prompt / encode_video（64 帧分块）/ question_answering
├── llava_onevision_rekv.py  # LLaVA-OneVision 适配（含 question_answering 的检索编排）
├── longva_rekv.py / video_llava_rekv.py / flash_vstream_rekv.py
└── longva/                  # LongVA 源码（含 trl，已打包进仓库）
video_qa/
├── rekv_stream_vqa.py       # 流式评测：按问题时间窗增量编码→问答
├── rekv_offline_vqa.py      # 离线评测
├── run_eval.py              # 多卡分片评测入口（num_chunks）
└── eval/                    # 多选/开放问答/MLVU/EgoSchema 评测脚本
```

关键流程（以 LLaVA-OneVision 为例）：

1. `encode_init_prompt`：system prompt 先过一遍模型，KV 留作 `n_init` sink；
2. `encode_video`：视频按 64 帧分块，`_encode_video_chunk` 逐块过 vision tower + multimodal projector + pooling（196 tokens/帧），经 patch 后的 attention 进入 `ContextManager.append`——滑窗内计算、**全量 KV 落盘/CPU**；
3. `question_answering`：先对所有 layer 的 kv_cache `set_retrieval()`；只喂 question 做一次前向触发 internal 检索（或直接设 external indices）；拿到检索后的 KV；`reset_retrieval()`；再喂完整 prompt 做 prefill + 自回归解码。

`patch.py` 的注释里有一句实诚话："This approach lacks scalability and will be refactored."——按模型类名逐个 if-else 打补丁（Llama/Mistral/Qwen2/MiniCPM），作者自己也知道这不可持续。

## 5. 评测与复现

```bash
bash prepare.sh                                    # conda 环境
# model_zoo/ 下放 HF 权重；data/ 下放 6 个 benchmark（见 README 目录结构）
sudo sysctl -w vm.max_map_count=262144             # offload 必需
python -m video_qa.run_eval --num_chunks 8 --model llava_ov_0.5b \
    --dataset qaego4d --sample_fps 0.5 --n_local 15000 --retrieve_size 64
```

- 模型：llava_ov_0.5b / 7b / 72b（72b 需 4×80GB，故 num_chunks 按 GPU 数折算）、video_llava_7b、longva_7b
- 数据集：qaego4d、egoschema、cgbench、mlvu、activitynet_qa、rvs_ego、rvs_movie
- 官方环境：Ubuntu 22.04、CUDA 12.6、8× H800 80GB；MLVU 有 ~9 小时视频，内存不够要在标注文件里删掉它
- `results/` 目录在仓库里是空的（评测产物不进仓库）

## 6. 方法论观察与批判性思考

**强的地方**：

1. **Training-free 即插即用**：不动权重、不训练，纯推理期改造——落地成本低，这是它相比重训长视频模型的最大优势。
2. **问题分解干净**：编码期（滑窗降本）/ 存储期（offload 保信息）/ 问答期（检索提效）三阶段正交，每阶段都可独立替换（比如 external retrieval 口子）。
3. **用模型自己做检索器**：internal retrieval 不引入外部 embedding 模型，避免了跨模型语义鸿沟；query/key 都是同一模型的表示，相似度天然对齐。
4. **工程完整**：triton/torch 双实现、异步流 overlap、LRU、pin_memory、多卡分片评测——是真的为大规模跑通而写的。

**可商榷 / 值得深想的地方**：

1. **检索质量的天花板**：block 代表 key 是 block 内 key 的均值（`_calc_block_topk` 里对 block_size 维 mean），粗粒度表示 + 余弦相似度，对细粒度问题（"第 37 分钟黑衣人说了什么"）可能不够。chunk 机制保连续但也会带入噪音 block。
2. **滑窗 + 检索的双重近似**：编码期滑窗丢了长程依赖（靠 offload 的全量 KV 补），问答期检索只取 top-k——两次近似叠加，误差如何累积论文里需要仔细看消融。
3. **位置信息的处理**：RoPE 换成 ESM、init token 同位置（Infinite-LM 式），检索回来的 block 在拼接时**失去了原始时间顺序的位置编码连续性**——模型看到的 KV 顺序是"init + 检索块按相似度/排序"，时间推理能力可能受影响。这是 RAG 式 KV 检索的固有问题。
4. **评测成本**：6 个 benchmark、5 个模型，官方用 8×H800；普通复现者只能跑 0.5b + 小数据集，结果外推要谨慎。
5. **代码可维护性**：monkey-patch + 按类名 if-else + "will be refactored" 的注释——能用，但接新模型（比如 Qwen3 系）要手动加分支。
6. **流式假设**：`rekv_stream_vqa.py` 里是按问题时间窗"编码到哪问到哪"，真正的实时流（视频边播边问、问题可回看）还需要外层调度，仓库只给了评测脚本。

## 7. 与其他工作的关系（笔记）

- **vs Flash-VStream**（致谢里引用）：同是流式视频理解，Flash-VStream 侧重特征压缩记忆，ReKV 侧重 KV-cache 的存取与检索——"压缩记忆 vs 外存检索"两条路线。
- **vs InfLLM / StreamingLLM**（致谢里引用）：滑窗 + sink 的思想直接继承，ReKV 的增量是把这套用到了**多模态 KV** 上并加了检索。
- **vs LeJudge**（同期 fork）：都是"给模型加约束/记忆而不重训"，一个在规划环路里加自然语言 cost，一个在推理期加可检索的 KV 外存——"不碰权重、只改推理期"是共同哲学。
- 潜在延伸：retrieval 的 block 粒度如果从"帧"变成"语义事件"，或检索器换成轻量双塔模型，可能是 follow-up 方向；`temporal_before` 式的时序约束（LeJudge 里有）在这里是缺失的一环。

## 8. 待探索

- [ ] 精读 ICLR 2025 论文（重点：检索消融 topk/chunk_size、长视频上的延迟-精度曲线、与全量 KV 的对比）
- [ ] 看 `model/attention/dot_production_attention/` 的 triton 实现，理解多阶段点积 attention 如何配合滑窗
- [ ] 跑通最小复现：llava_ov_0.5b + qaego4d 子集（需 GPU，当前 VM 无卡，先记）
- [ ] 检查上游 issue 区是否有已知 bug（如 position 处理、72b 分片）

---

*本文档为个人学习笔记，基于 2026-10-07 对上游 `main`（`1fd9a3d`）的代码走读 + README。论文观点均为转述。*
