# vllm-fastdecode（缩小版原型）

给 vLLM **v0.31.0** 加一层 CPU 上的 KV cache，并让 CPU 就地计算这部分 attention，参考 [vllm#3606](https://github.com/vllm-project/vllm/issues/3606) / FastDecode（arXiv 2403.11421）和 NEO（MLSys'25）。以 vLLM 插件形式实现，**不改 vLLM 源码，不用编译 CUDA**。

## 原理和这版的取舍

- 显存只能放少量 KV，所以 batch 开不大。这里再加 C 个放在内存里的 KV block，调度器看到的是 G+C 个 block（G 为显存中的 block 数）。
- block 按编号分层：`id < G` 的在显存，`id ≥ G` 的在内存。分配时**优先用显存 block**，显存满了才用内存 block。
- decode 时，一个请求的 attention 拆成两半：GPU 部分（FlashAttention 算显存里的 block）和 CPU 部分（C++/AVX2 kernel 算内存里的 block），最后用 log-sum-exp 合并。**数学上与原始 attention 等价**，区别只在浮点舍入。
- 纯 prefill 照常在 GPU 上，用当前步的 q/k/v 计算。
- 某一步里如果没有请求用到 CPU block，就**完全走原生 FlashAttention 路径**。

这一版**没做**的：GPU/CPU 流水线重叠、负载稳定调度、跨节点 R-worker、TP>1、prefix caching、chunked prefill、speculative decoding、滑动窗口和 MLA 模型、FP8 KV。配置里打开不支持的选项时，启动会直接报错，不会悄悄算错。

## 安装

要求：原生 Linux（WSL2 的 pinned memory 受限，拷贝会变慢）；NVIDIA 驱动支持 CUDA 13（PyPI 上的 vLLM 0.31.0 wheel 基于 torch 2.13+cu130）；g++ ≥ 9；Python 3.10–3.13。

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install vllm==0.31.0 ninja datasets scipy pytest
uv pip install -e . --no-deps
python -m pytest tests -q   # 只用 CPU；第一次运行会编译 kernel，约 30 秒
```

用 Docker 或 Apptainer（Torch 集群）：

```bash
docker build -t fastdecode .
docker run --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface -it fastdecode bash
```

在 Torch 上用 `apptainer build fastdecode.sif docker-daemon://fastdecode:latest`，或把镜像推到 registry 后再 `apptainer pull`。

> 这个插件依赖 vLLM 的内部接口（`EngineCore._initialize_kv_caches`、`FreeKVCacheBlockQueue.prepend_n`、FlashAttention backend 的内部实现），所以**版本锁定在 0.31.0**。PyPI 上已经有 0.32.0，不要直接升级。

## 使用

```bash
FASTDECODE_CPU_KV_GB=32 FASTDECODE_CPU_THREADS=6 \
vllm serve lmsys/vicuna-7b-v1.5 --attention-backend CUSTOM \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-batched-tokens 4096 --max-num-seqs 128
```

在 Python 里用：`LLM(..., attention_backend="CUSTOM", enable_prefix_caching=False, enable_chunked_prefill=False)`，环境变量同上。

| 环境变量 | 含义 |
|---|---|
| `FASTDECODE_CPU_KV_GB` / `FASTDECODE_CPU_BLOCKS` | CPU KV 层的大小（二选一）。不设置时，CUSTOM backend 等同于原生 FlashAttention |
| `FASTDECODE_CPU_THREADS` | CPU kernel 的线程数，建议设为 P 核数 |
| `FASTDECODE_MARCH` | JIT 编译用的 `-march`，默认 `native` |
| `FASTDECODE_STATS=1` | 每 50 步打印一次请求在 GPU/CPU 上的分布 |
| `FASTDECODE_VERIFY=1` | **调试用，很慢**：在线自检（见下文"排查正确性问题"） |

## 实验流程（按顺序跑）

`scripts/run_all.sh` 会把下面几步串起来。

1. **`pytest tests`**：CPU kernel、规划器、整个 split/merge 前向流程（GPU 算子用参考实现替代）、block 分配策略、启动时的容量检查、在线自检，共 51 个测试。
   **`scripts/gpu_selftest.py`**：同一个前向流程换成**真实的 CUDA 算子**（FlashAttention、reshape_and_cache_flash、merge_attn_states），不加载模型，模拟连续批处理（请求到达、decode、结束、block 复用），每一步都和"把全部 KV 放在一块大显存里的原生 FlashAttention"逐元素对比。约 1 分钟。
2. **`scripts/probe_cpu_attn.py`**：测出本机 CPU 每秒最多能贡献多少 token。**先跑这一步**，它决定这台机器能不能看到加速（见下一节）。
3. **`scripts/check_correctness.py --task tokens`**：故意把 GPU KV 设得很小，强迫大部分 KV 放到 CPU，然后与原生 vLLM 逐 token 对比 greedy 输出，判定标准与 vLLM 自己的测试一致（`check_logprobs_close`）。同时测**噪声底**：原生 vLLM 只改 `max_num_seqs` 时自己和自己的差异。vLLM 不是 batch-invariant 的，而 FastDecode 必然会改变 batch 构成。
4. **`scripts/bench.py`**：所有组的 GPU KV 大小（`--gpu-kv-gib`）完全相同，fastdecode 组只额外多一个 CPU 层。每个 session 启动一次新引擎，按 ABBA 顺序交替，每组至少 4 个 session。注意原生 vLLM 要求 GPU KV 至少能放下一条 `max_model_len` 的请求（FastDecode 组会把 CPU 层算进去），所以脚本默认 `--max-model-len 1024`。
5. **`scripts/analyze.py`**：以 session 为统计单位，给出吞吐比的 bootstrap 95% CI、Welch t 检验、Mann-Whitney 检验，并报告 TPOT。
6. （可选）**`check_correctness.py --task gsm8k`**：任务级正确性。用 McNemar 检验看是否有差异，用 **TOST 等价检验**支撑"没有损失"的结论。**"不显著"不等于"等价"**，只有 TOST 才能支撑这个说法；margin 要事先定好。

可选的实验组：`baseline`（原生，默认 cudagraph）、`baseline_pw`（原生，只用 piecewise cudagraph，这是 FastDecode 强制使用的模式）、`fd_c0`（走 FastDecode 代码路径但 CPU 层为 0，用来测额外开销，应该约等于 `baseline_pw`）、`fastdecode`。

## 排查正确性问题

`check_correctness.py` 报 `CHECK` 时按顺序做：

1. `python scripts/gpu_selftest.py`：真实 CUDA 算子下 split/merge 是否正确。FAIL 说明问题在 backend 的数值路径。
2. 只重跑 fastdecode 组并打开在线自检（`--reuse` 复用已有的两个 baseline 组结果）：
   ```bash
   FASTDECODE_VERIFY=1 python scripts/check_correctness.py --task tokens \
     --gpu-kv-gib 0.25 --cpu-kv-gib 8 --cpu-threads 10 --limit 128 \
     --reuse baseline,baseline_alt 2>&1 | tee verify.log
   grep "VERIFY" verify.log | head -60
   ```
   自检完全基于 vLLM 自己的 GPU 端元数据（block table、seq_lens、slot_mapping），不依赖 FastDecode 的执行计划。它检查：执行计划用的 seq_lens 是否与 GPU 一致；每个被 attend 的位置上存的 KV 是否就是最后一次写给这个位置的（能发现写丢、写错位置、block 还在用就被复用）；输出是否与 fp32 参考一致。全 GPU 请求（P 类）的误差就是原生 FlashAttention 的噪声水平，可以作为对照。`FASTDECODE_VERIFY_LAYERS=all` 检查所有层（更慢）。
3. 自检没有任何 violation 但仍然 FAIL：问题不在 attention。用 `--arms baseline,baseline_pw,fd_c0 --reuse baseline` 分别看 piecewise cudagraph 和 CUSTOM backend 本身（不带 CPU 层）是否已经造成偏差。

## 什么时候会快

这一版是同步执行：每一步的耗时约等于 GPU 时间加上 CPU attention 时间。只有当 **CPU 每秒能多处理的 token 数大于 baseline 的吞吐**时，加上 CPU 层才会更快：

- CPU 上限约等于 CPU 内存带宽 ÷（每 token 的 KV 字节数 × 平均上下文长度）。`probe_cpu_attn.py` 会直接打印这个数。
- baseline 吞吐约等于（显存能放的 token 数 ÷ 平均上下文长度）÷ 每步耗时。

以笔记本 3080 Ti（16GB）跑 vicuna-7b fp16 为例：权重 12.6 GiB，大约只剩 1 GiB 给 KV，也就是 2048 个 token，baseline 只能同时跑 3–4 条序列，所以这正是 FastDecode 最有利的情形。如果 probe 显示 CPU 上限低于 baseline，同步版就会更慢，这时需要做第二阶段的流水线重叠（见下）。

另外，这是**用延迟换吞吐**：TPOT 一定会变高，analyze.py 会一并报告。

## 文件

```
fastdecode/__init__.py   插件入口：注册 CUSTOM backend，按需打 patch
fastdecode/patches.py    调度器看到 G+C 个 block；空闲队列中显存 block 优先
fastdecode/backend.py    Backend / MetadataBuilder / Impl（GPU 与 CPU 分算 + 合并）
fastdecode/plan.py       每步的执行计划（纯 CPU 张量，可单独测试）
fastdecode/state.py      worker 进程里的 CPU KV 存储和 pinned 缓冲区
fastdecode/csrc/cpu_attn.cpp  CPU kernel（OpenMP + AVX2/F16C，fp32 累加）
fastdecode/verify.py     FASTDECODE_VERIFY=1 的在线自检
scripts/                 probe / gpu_selftest / check_correctness / bench / analyze / run_all.sh
tests/                   只用 CPU 的测试
```

## 第二阶段（如果有时间）

1. **流水线重叠**：参考 vLLM 的 DBO（`--enable-dbo`）把 batch 切成两个 micro-batch，让一个 micro-batch 的 CPU attention 和另一个的 GPU 计算并行。
2. 论文 Algorithm 1 的负载稳定调度。
3. 支持 chunked prefill 续算：把 CPU 上的那部分前缀拷回 GPU，或者在 CPU 上算 prefill 的那部分。
