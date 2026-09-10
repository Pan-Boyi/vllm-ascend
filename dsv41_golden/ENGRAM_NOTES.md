# Engram：官方实现调研与我们的路线 · v1.0

判据 **G-3**（查表逐位相同）和 engram host offload 的技术底账。
所有结论附文件路径 + 行号，`engram_research/raw/` 下有全部原始源码可复核。

---

## 0. 一句话

**`ElasticBuffer` 的 engram 路径是字节透明的（已实测），所以 fp8 字节可以装进 bf16 外衣
存 203 GB 而不是 393 GB。但它的远端（URMA/RDMA）取行在本机不可用（kernel 内断言），
所以分片表这条官方形态走不通。**

| 结论 | 强度 |
|---|---|
| fp8 字节装 bf16 外衣不被解释 | ✅ **已实测确认**（源码 3 层 + 位保真 1 层，见第 3.1d 节）|
| 本地 host gather 可用 | ✅ 已实测 |
| 远端 RDMA 取行可用 | ❌ **已实测证伪**（第 3.1c 节）|
| 分片表的 engram offload | ❌ 被上一条挡住 |

官方 host 存储的 dtype 白名单里没有 fp8（第 3.2 节），技报描述的 fp8 路**没有开源**；
开源的是 bf16 变体（≈393 GB，**且是 pinned 不可回收**）。

---

## 1. G-3 的前提已实测验证：可以要求逐位相同

查表就三行：
```python
row = table[hash_id]                          # [256] float8_e4m3fn
sc  = scale[hash_id]                          # [8]   float8_e8m0fnu，每 32 个值一个
out = (row.float() * sc.float()).bfloat16()
```

没有归约、没有累加顺序可争论 —— 所以**不像协议里其他任何张量，这一个没有浮点误差借口**。

**而且这个前提是验证过的，不是假设的**（`engram_lookup_ref.py`）：
- `e8m0` 是纯指数 → scale **恰好是 2 的幂**；`e4m3` 有 3 位尾数
  → **乘积在 fp32 里精确可表示，乘法不产生任何舍入**。
- 实测：把反量化改用 fp64 重做，**bf16 结果一个 bit 都不差**（两层、全部 648 行）。
- 整个操作里**只有最后 fp32→bf16 那一次舍入**，两边做法相同。

⚠️ **适用边界**：这是对**实际用到的 648 行**验证的，不是对全部 3.84 亿行的证明。
论证是结构性的（指数 × 3 位尾数），我认为对全表成立，但只测了用到的那些。

### 参考产出

```bash
python engram_hash_ref.py   --verify $DU/full40_v2 --decode-from 20   # 整数哈希，9 秒，不加载权重
python engram_lookup_ref.py --out $DU/engram_lookup_ref               # 查表结果
```

| | 指纹 |
|---|---|
| `engram_hash_ids` `[1,27,2,24]` int64（dim=2 顺序 = **[layer 1, layer 14]**）| sha `39fa11acd920fafd` |
| layer 1 查表结果 `[27,24,256]` bf16 | sha `132146ac8219dd43` |
| layer 14 查表结果 `[27,24,256]` bf16 | sha `ca89820962d7bbfa` |

实测表规模：`layers.1.engram.embed.weight [384006168, 256]` float8_e4m3fn
+ scale `[.., 8]` float8_e8m0fnu；layer 14 是 `[384016682, 256]`。**1D block-32。**

---

## 2. 两份官方实现的哈希是同一个（AST 级对比）

`/workspace/models/DeepSeek-V4.1-Flash/inference/engram.py`（我的 golden 跑的，逐字副本在 `ref/`）
vs `engram_research/raw/dsv41/models/modules/engram.py`（CANN day-0）：

| 函数 | 结果 |
|---|---|
| 滚动 XOR 哈希本体 | **逐字相同**（连注释都一样）|
| `compute_hash_multipliers` | **AST 完全相同** |
| `EngramLayout` + `from_args`（素数桶、offsets） | **AST 完全相同** |
| `find_next_prime` | 循环结构不同，语义相同 |
| `build_compressed_token_map` | 只差一行 import 的位置 |

**⇒ 上面那些参考指纹对 CANN 那份实现同样成立。**

### ⚠️ 但回看（look-back）机制的结构不同

| | 模型自带（golden 跑的） | CANN day-0 |
|---|---|---|
| 历史来源 | **内部 `cache` buffer**，`gather(positions - shift)` | **调用方传 `prefix_input_ids`** |
| prefill 的 shift | 同一个 cache gather | `F.pad(x, (k,0), value=pad_id)[:, :T]` |
| 截断标记 | `blocked \|= (positions < shift) \| (source == DEAD)` —— **累积** | `torch.where(ngram_shift_mask, ...)` —— **逐元素**，累积性由调用方保证 |

**纯文本路径两者等价**：没有 DEAD 时只剩 `positions < shift`，而它对 shift 本来就单调，
所以累积与逐元素同结果。**标准 27 token 输入上能逐位对上。**

🔴 **image span（DEAD）上可能分叉**：模型自带那份一旦在 shift 2 撞到 DEAD、shift 3 也一律 pad；
CANN 那份的累积性**必须由调用方的 `ngram_shift_mask` 保证**。mask 不累积的话，
一个 n-gram 会「跨过」image span 去取更早的真 token。
⚠️ 这条是**从代码结构推的，未实测**（纯文本跑不到 DEAD），vision 路径本就在范围外。

🔴 **decode 的边界条件是 CANN 那条路独有的**：
```python
if prefix is None:  raise ValueError("prefix_input_ids can not be None on the decoding phase.")
start = prefix.shape[1] - k
if start < 0:       raise ValueError("prefix_input_ids length ... is insufficient for shift ...")
```
`max_ngram_size = 4` ⇒ decode 时 `prefix_input_ids` 至少要 3 个 token，
序列刚开始只有 1~2 个时会撞第二个 raise。**我的 `--decode-from` 测试覆盖不到这个**
（它验的是 cache 语义，`positions < shift` 自动兜住）。走显式 prefix 的实现要自己测
prefix 长度 = 3 / 2 / 0 三种。

🔴 **CANN 把一个 assert 注释掉了，模型自带那份是打开的 —— 建议打开**：
```python
# raw/dsv41/models/modules/engram.py:165-166
# Add this assert check after adopting the new tokenizer.
# assert vocab_size == config.engram_compressed_vocab_size, ...
```
所有哈希乘子都从压缩词表大小推导，**vocab_size 对不上 → 整张表静默 rehash**，
查出来全错但数值看起来完全合理（都是正整数、都在表范围内）。
实测我们的 checkpoint 是 `compressed_vocab = 99092`，与 config 一致。

---

## 3. `ElasticBuffer`：官方 host offload 算子

### 3.1 机制

- **推理模式 = RDMA one-sided read**。host pinned 段由 `aclrtMallocHost` 分配、经
  `aclrtHostRegisterV2` 映射出 device 可访问地址；本 rank 命中直读本地 pinned，
  远端 rank 走 RDMA（`ElasticBuffer.md:142, 178, 196`）。底层是 URMA→UBMem
  （`tech_report.md:200-210`），声称「不占用计算核」。
- API 异步：`engram_fetch(indices)` 立即返回一个 callable，执行它才阻塞
  （`ElasticBuffer.md:234`）。**但官方参考实现里 fetch 之后立刻 wait，零 overlap**
  （`raw/dsv41/models/modules/engram.py:427-433`）。技报声称的多流 overlap 不在开源代码里。
- `num_cpu_bytes` 必须 **2 MB 对齐**，每条目 32 字节对齐（`ElasticBuffer.md:310-317`）。
  我们 `hidden=256`：bf16 = 512 B/条，fp8 = 256 B/条，都天然 32 对齐 —— **对齐不是障碍，dtype 是。**

### 3.1b 🔴 实现前提：`engram_write` 之前必须先跑过一次真实集合通信

**HCCL 通道是惰性建立的。** `init_process_group` 不足以建通道，
**`ElasticBuffer` 的构造也不足以**（它成功返回、还给出了 comm name）——
但 `engram_write` 需要通道已经存在。

实测（卡 2+3，2 rank）：

```
init_process_group → ElasticBuffer(...) → engram_write
    ❌ RuntimeError: Acquire HCCL channel failed, ret: 11 (rank0) / ret: 4 (rank1)
       at elastic_buffer.py:529  self._runtime.engram_write(storage)

init_process_group → all_reduce → ElasticBuffer(...) → engram_write
    ✅ engram_write OK        ✅ engram_fetch OK -> (4, 128) bfloat16
```

**唯一的变化就是那一次 `all_reduce`。**

**⇒ 正确的初始化序列：**
```
init_process_group  →  ★至少一次真实集合通信★  →  ElasticBuffer(...)  →  engram_write
```

⚠️ **这在真实推理里大概率自然满足**（起服务时早就跑过集合通信），
**但在单元测试和最小复现里会踩** —— 那些环境往往只 init 不通信。
所以实现里建议**显式做一次**，而不是依赖调用环境凑巧满足。

#### 定位过程（留给下一个碰到它的人）

逐步探针（每步一个 `print(flush=True)`）：
```
1 import torch_npu ✅   2 set_device ✅   3 init_process_group ✅   4 WORLD group ✅
5 import ElasticBuffer ✅   6 size_hint=2097152 ✅   7 get backend ✅
8 hccl_comm_name='group_name_0' ✅   9 ElasticBuffer 构造 ✅   10 CPU storage ✅
11 engram_write ❌
```
**不是 dtype、不是 128 对齐、不是构造参数、不是 world_size** —— 那些都过了。

⚠️ **第一次跑时的错误签名是误导性的**：`vector core exception`（由
`torch_npu/distributed/distributed_c10d.py:500` 的 watchdog 在**拆解路径**上报），
它会把人引向「AIV 算子故障」。真错是 `Acquire HCCL channel failed`。
**两个签名来自同一个故障。**

⚠️ 而且第一次跑的真错**被我自己的 `| grep ... | head -40` 吃掉了** ——
`head` 不 flush，攒不够行数就不输出，进程结束时全丢，日志 0 字节。
**当时我据此判断「它死在第一行 print 之前」，那是工具假象，不是事实。**
教训：**过滤器本身可以吃掉证据，而吃掉之后看起来和「证据不存在」一模一样。**
诊断故障时不要给输出接 `head`。

#### 热身之后：`engram_write` + `engram_fetch` 本地路径通了，数值正确

```
[r0] constructed  comm_buffer_size=0 rank_size=0
[r0] engram_write + synchronize OK
[r0] fetch returned (4, 128) bfloat16 -- NOT yet synchronized
[r0] synchronize after fetch OK          ← 真正的判据
[r0] D2H OK  first row[:4] = [0.0, 1.0, 2.0, 3.0]   ← 源是 arange 填的，值对上了
```
两个 rank 都成功，`num_max_tokens_per_rank` 设不设都成功（它是 MoE dispatch 的参数，
engram 路径不用；`comm_buffer_size` 全程是 0 也不影响）。

🔴 **判「NPU 算子跑通了」必须先 `torch.npu.synchronize()`。**
AIV 是异步的，错误要到下一个同步点才浮现。我曾经拿到 `engram_fetch` 的返回值就报了
「OK」—— **那是假通过**，那次的 kernel 其实 trap 了，只是我从没同步过。
「函数返回了」和「计算成功了」是两件事，而失败会伪装成成功。

**⇒ 探针里把「fetch 返回」和「同步后仍然 OK」打印成两条不同的消息**，避免自己再读错。

#### 已排除的（都实测过，不要重试）

| 假设 | 结论 |
|---|---|
| UB/URMA 需单独启用或配置 | ❌ 不需要 —— **零 HCCL 环境变量**下通了 |
| `num_cpu_bytes` 2 MiB 太小 | ❌ 够 —— 就是 `get_engram_storage_size_hint` 自己算的 2097152 |
| 需要调用方 barrier | ❌ 不需要 —— 文档 `:158` 那个 `Barrier → write → Barrier` 是**算子内部做的**（`elastic_buffer.cpp:943/965` 两处 `EngramBarrier(false, true)`）|
| `ulimit -l` 锁页限制 | ❌ 已是 `unlimited` |
| CANN 9.1/9.2 混搭 | ❌ 三个独立来源否掉：本测试的 allreduce、integration 的 EP all2all、kernel 的 9.2 单卡算子 |
| `nnal/atb/set_env.sh` 没 source | ❌ 不是 HCCL 依赖（`env_dsv41.sh` 刻意剥掉了 nnal，因为 ATB 是对着 9.1 编的）|

### 3.1c 🔴🔴 远端（URMA/RDMA）取行在本机不可用 —— kernel 内部断言失败

**这是目前挡住 engram offload 的硬阻塞。**

同一个 `engram_fetch` 调用，只改 indices 落在本地还是远端（卡 2+3，2 rank，同 UB 平面）：

| indices | 结果 |
|---|---|
| **只含本 rank 的行** | ✅ 两个 rank 都成功，**数据正确** |
| **只含远端 rank 的行** | ❌ 两个 rank 都挂 |
| **本地 + 远端混合** | ❌ 两个 rank 都挂 —— **没有部分成功，整次调用失败** |

```
[AIV Block 1/56] [ASSERT]
  .../ops-transformer/build/binary/ascend950/src/engram_fetch_wait/op_kernel/arch35/
  engram_fetch_wait_arch35.h:99
  void Mc2Kernel::EngramFetchWaitArch35::Process(): Assertion `ret == 0' failed.
  Urma drain failed, ret=255, globalChannelIdx=1
→ RuntimeError: npuSynchronizeDevice ... device error type 3, error code 507035
```

**`Urma drain failed`** 就是技报 `:200-210` 描述的那条 **URMA→UBMem** 通路
（`aclrtHostRegisterV2` 映射 + RDMA one-sided read），在**闭源 kernel 内部**失败。
`ret=255` 的含义查不到。

#### ⇒ 分片的 engram 表在本机走不通

表按 `engram_tp_size == world_size` 分片，**任何 hash id 都可能落在远端 rank**
—— 远端取行不是可选优化，是必需路径。

| 布局 | 可行性 |
|---|---|
| 表分片到 N 个 rank（官方设计） | ❌ 远端取行必需，而它挂 |
| 整表压一个 rank、其余持 0 行（`num_entries=0` 官方支持） | ⚠️ **只有持表的 rank 能本地取**，其余 rank 取任何行都是远端 → 同样挂。除非只让持表的 rank 查表、结果用普通集合通信分发 —— 那 `ElasticBuffer` 只剩「本地 host gather」一个功能 |
| 每个 rank 持完整一份 | ❌ 203 GB × 4 = 812 GB pinned，机器只有 755 GB |

#### 兜底路径

`vllm_ascend/attention/sfa_kv_offload.py`（1156 行）：显式 D2H/H2D + 常驻 topk 缓冲，
**不依赖 URMA**，访问形态与 engram 同构。**兜底是「改造现成机制」而不是「从零写 gather」。**

⚠️ 注意「fp8 字节能不能装进 bf16 外衣」（AD2 的核心问题）**与这条独立** ——
兜底路径同样要面对存 fp8 还是 bf16 的选择，所以那个结论无论如何都有用。

### 3.1d ✅ E-ENG-1：字节透明性已实测确认（本地路径）

`e_eng_1_bit_fidelity.py`，卡 2+3，2 rank：

```
cann_ops_transformer from .../cann-9.2.0-beta.2/...          ← 树断言通过
HCCL warm-up all_reduce -> 3.0                               ← 前提 1
128-alignment check: 128 % 128 == 0                          ← 前提 2（压线）
storage carries 2080 NaN and 16 Inf bf16 values              ← 刻意的病态数据
engram_write OK
engram_fetch OK -> (5, 128) bfloat16 (as bytes (5, 256))
  ok idx 0 / 1 / 2 / 3 / 4095   local   byte-identical
local rows byte-identical : 5/5
hostile-pattern row       : 1/1          <-- NaN/Inf/denormal 字节模式
E-ENG-1 byte fidelity: PASS               exit=0（加了 destroy() 之后干净退出）
```

**为什么这个 PASS 强**：storage 里刻意埋了 **2080 个 NaN + 16 个 Inf**（读作 bf16）——
+Inf / -Inf / quiet NaN / signalling NaN / 全尾数 NaN / 负零 / 最小最大 denormal /
最小 normal / 最大有限 / 交替位，另加一整行 0..255 全字节值。
**任何 dtype-aware 的处理（NaN 规范化、denormal 冲零、值域检查）只会破坏这些、
而且只破坏这些** —— 正常数值全测一遍也看不出来。**它们全部逐位存活。**

🔴 **比对按字节（`.view(torch.uint8)`），绝不按值。** NaN ≠ NaN，按值比会
**恰好在最有意思的地方假通过**。（反向也成立：我在一个快速探针里用 fp64 算期望值，
bf16 在 1000 附近 ULP = 4 装不下 1001/1002/1003，**正确的结果被判成失败** ——
见第 5 节的坑列表。）

**⇒ 加上源码三层证据**（`elastic_buffer.py` 的 bf16 白名单、`elastic_buffer.cpp` 的裸
`uint8_t*` memcpy 且 `engramDtype_` 只写不读、`EngramFetch` 里 `dtypeEnum` **根本没传给**
`aclnnEngramFetch`），**四层一致：写入、取回、AIV kernel 都不解释数值。**

### 3.1e 🔴 三个实现前提（缺一个就挂，且症状都误导）

```
init_process_group
  → ★1. 至少一次真实集合通信★        缺 → Acquire HCCL channel failed, ret 11/4
  → ElasticBuffer(...)                （★2. hidden % 128 == 0，压线无余量★）
  → engram_write → engram_fetch → ...
  → ★3. eb.destroy()★                 缺 → 两个 rank 都 SIGSEGV（exitcode -11）
  → destroy_process_group
```

**前提 3 的症状最误导**：崩在**所有工作之后** —— 判决已完整算出并写盘，
但 torchrun 把一个通过的测试标成 `FAILED`，而崩溃看起来像是测量本身出的问题。
`ElasticBuffer` 持着 host pinned 内存和 HCCL 内存句柄（`destroy()` 在
`elastic_buffer.py:830`），把 process group 从它脚下抽走就是这个 segfault。

🔴 **教训：进程退出码不是判据。** 一个在结论落盘之后崩溃的测试，和一个失败的测试，
**退出码一模一样**。我当时是靠 JSON 文件才分辨出来的 —— 如果只打印到 stdout，
SIGSEGV 会连输出一起吞掉，就与真失败完全不可区分。
**⇒ 判据必须落盘，而且要落在崩溃可能发生的地方之前。**

### 3.1f 访问模式实测：去重在 decode 里收益为零（构造决定）

用 `engram_hash_ref.py`（9 秒、不加载权重）测 layer 1 的 id：

| 文本 | token | ids/层 | unique | **去重收益** | index/行 字节比 |
|---|---|---|---|---|---|
| golden prompt | 27 | 648 | 648 | **0.0%** | **1.6%** |
| 自然英文散文 | 52 | 1248 | 1224 | **1.9%** | **1.6%** |
| 高度重复文本 | 281 | 6744 | 240 | 96.4% | 43.9% |

🔴 **单个 token 的 24 个 id 必然互不相同**，实测 27/27 个 token 都是 24 个不同 id。
原因是结构性的：24 列 = 3 个 n-gram 尺寸 × 8 head，**每列有自己的素数桶区间且互不重叠**
（`offsets` 是各列素数的累积和；实测两两区间无重叠）。

**⇒ decode 每步只有 1 个 token → 去重买到的是零，官方的 `unique_local_entry` 也一样。**
**⇒ 自然文本的 prefill 上去重收益 0~2%；只有病态重复文本才有 96%。**
**⇒ 索引交换只占数据量的 1.6%** —— 24 个 int32 = 96 B vs 24 行 × 256 B = 6 KB。

对「自己用 all_to_all 替代 URMA」这个形态（AD2'）的意义：**不去重的代价是 0~2%，
不是一个数量级**；而每步每层的数据量是 6 KB、两层 12 KB，**极小消息** ——
**所以成本主项会是 a2a 的固定延迟 × 4 次，不是带宽。** 该测的是延迟不是吞吐。

### 3.2 🔴 两条挡住我们的硬约束

**(a) host 存储不支持 fp8。** `ElasticBuffer.md:477`（「Engram dtype约束」）：
> `storage.dtype` 仅支持 `bfloat16`、`float16`、`float32`。

代码里还有一个显式守卫（`engram.py:369-373`）：
```python
if self.weight.dtype != torch.bfloat16:
    raise TypeError("ElasticBuffer only accepts BF16 Engram embeddings, ...")
```

**注意这不是算子库的能力缺失** —— 同一份文档的 `dispatch` 接口**是**支持 fp8 的
（`:370`：「支持 `bfloat16`、`float16`、`float8_e5m2`、`float8_e4m3fn`」，还带 scales 入参）。
**fp8 缺失是 engram 这条路径的选择。**

**(b) engram 通信域 `world_size ∈ [2, 1024]`。** `ElasticBuffer.md:521`（「通信域约束」，
确认是 engram 那条不是 EP 那条）。**单卡 world_size=1 不在支持面内。**
而 `modeling_deepseek.py:1640-1655` 在 `enable_engram_offload=True` 时强制
`engram_tp_size == world_size`，否则 raise。

> 可能的绕法：`ElasticBuffer.md:524` 明确写「支持 `num_entries = 0`」，
> 所以理论上可以构造 world_size=2、一侧 0 行的域。**未验证**，而且要两张卡。

### 3.3 官方存 bf16，不是 fp8

`engram.py:292-301` offload 模式下 `params_dtype` 硬定为 `torch.bfloat16`；
`:349-359` 在 CPU 上把整个 shard `fp8 → fp32 → × scale → bf16`；
`:374-385` `engram_write(self.weight)` 然后 `del self.weight`。

**体积** = (384008192+384018432) × 256 × 2 B ≈ **393 GB（366 GiB）**。

**技报和开源代码在这一点上互相矛盾**，两边都核实过：
- `tech_report.md:69`：「**Engram 表**：大表保持 FP8 存储，查询时按行反量化到 BF16。」
- `tech_report.md:190`：「两表本体 2×384M×256×1B ≈ 197 GB…采用 mxfp8 量化」

**⇒ 技报描述的就是我们要走的路，但它没开源。**

### 3.4 ✅ 一条可以照搬的好消息：scale 常驻 HBM

`tech_report.md:212`：
> 量化的 scaling factor 参数数据量要小很多，当前没有 offload 到 host 侧，
> 全量 engram table 的 scaling factor 参数都是放在 device 的 local memory 中，由 AIV 进行 gather 到输出上。

scale 是 1D block-32，每行 `256/32 = 8` B → **两表合计 ≈ 6.1 GB**，单卡 A5 放得下。
**「scale 留 HBM、只从 host gather fp8 行」这个分工可行，传输量约为 bf16 方案的一半。**

### 3.5 没有任何 Triton / 自定义 kernel 可抄

整个 `raw/dsv41/` 里没有 `import triton`、没有 `@triton.jit`、没有 `.cu`/`.cpp`。
gather 全在 `ElasticBuffer` 这个**闭源二进制**里（AIV 实现）。
**⇒ 「Triton gather kernel 能否在 A5 编过」这个首周问题，官方零参考。**

---

## 4. 三个选项（待拍板：单卡还是允许 2 卡）

| 选项 | 做法 | 代价 | 风险 |
|---|---|---|---|
| **A. 官方算子 + bf16** | 直接抄官方那条路 | host **393 GB**；**需要 ≥2 卡** | 低（有可运行实现），但和「单卡」冲突 |
| **B. 自己写 fp8 host 查表** | 189 GiB fp8 + scale 留 HBM，自己写 gather | 3~5 人天，Triton-on-A5 未探明 | 中高，**零官方参考** |
| **C. 混合** | 抄算法部分，自己写存储 + gather | 同 B，少踩算法坑 | 同 B |

**倾向 C，但取决于「单卡 vs 2 卡」这个前提：**
- 单卡 → A 直接排除（world_size ≥ 2），必须 B/C。
- 允许 2 卡 → A 变成**有官方实现兜底**的选项，风险低得多。

**注意 TP=1 不等于 world_size=1**（可以 world_size=2 + attn TP=1），这个区别现在有实际后果。

---

## 5. 能直接抄的 / 必须自己写的 / 明确的坑

### ✅ 能直接抄（框架无关的纯算法，Apache-2.0，与存储方案无关）
- `engram.py:20-63` `build_compressed_token_map`（normalizer 链，含 `` sentinel）
- `engram.py:66-85` `compute_hash_multipliers`（奇数乘子 + int64 防溢出）
- `engram.py:88-136` `find_next_prime` / `EngramLayout.from_args`（素数桶、offsets）
- `engram.py:177-248` 滚动 XOR、prefix/decode 两种 shift、DEAD 屏蔽
- `engram.py:490-521` `Engram.forward`（wkv split、带符号 sqrt + sigmoid 门控、image_mask 直通）
- `modeling_deepseek.py:2214-2236` **「weight + scale 配对劫持 loader」的 pattern** ——
  对我们同样适用，只是配对之后**不反量化**，直接把 fp8 和 scale 分别落到各自存储

### ❌ 必须自己写
1. **fp8 host 表本身**（算子白名单锁死 bf16）
2. **gather kernel**（无任何源码可抄）
3. **运行期反量化**（官方是加载期在 host 做完，我们要的「gather fp8 → 卡上反量化」完全空白）
4. **单卡通路**（绕过 `ElasticBuffer`，或伪造 world_size=2）
5. **fail-safe**（越界 index 静默返回 0，见下）

### ⚠️ 明确的坑
- 🔴 **越界 index 静默返回 0，不报错**（`ElasticBuffer.md:230`）。查表判据是「逐位相同」，
  静默出 0 会表现为「某些行全零但看起来合理」。
  **必须在自己这层显式 range check 并拒绝执行。**
- 🔴 **官方的 `fp8_weight_loader` 不分块**（`engram.py:354-358`）：一次性把整个 shard
  提到 fp32 = 48M×256×4 B ≈ **49 GB 临时 host 内存**，多 rank 并发峰值极高。**照抄会炸。**
- **同一 `ElasticBuffer` 实例不允许并发 fetch**（`ElasticBuffer.md:486`）。
- **不支持投机解码**（`engram.py:243`：`input_ids = input_ids.view(-1, 1) # not support speculative decoding`）。
- **offload 模式下 TP 的 allgather/reduce_scatter 分支被显式跳过**（`engram.py:401-405`）。
- **别把技报当代码**：技报 `:69/:190` 说 FP8，开源代码 `engram.py:292-301,349-359` 是 BF16。

### ❓ 无法确定，但抄之前必须实测
`self.weight` 在 offload 之前**到底在 CPU 还是 NPU 上**，代码注释和算子文档**互相矛盾**：
- `engram.py:382-384` 注释：「ElasticBuffer's device-side write API consumes an NPU tensor」
- `ElasticBuffer.md:481`：「`storage` 必须在CPU上。」

原因是 `VocabParallelEmbedding` 的参数分配逻辑不在这份拷贝里
（`raw/dsv41/` 只有 `models/ utils/ config/`）。
**如果参数真在 NPU 上，offload 前每 rank 每层要在 HBM 里驻留 48M×256×2 B ≈ 24.6 GB，
两层 ≈ 49 GB 峰值。**

---

## 6. 分工（已定，`dsv4.1_flash_dev` 确认）

| 谁 | 做什么 |
|---|---|
| **model 线** | engram 主干（`wkv` 投影、`q_weight`/`k_weight`、写回 4 份残差流）—— 普通算子，和 attention 同类 |
| **golden 线** | 189 GiB fp8 表的 host 查表 / gather |
| 接口 | `hash_ids [B,L,24]` → `[B,L,24,256] bf16` |

这样切的好处是**判据分给两个人、边界干净**：查表按**逐位相同**判（本文第 1 节），
主干按逐层 rel_l2 判（`ALIGNMENT_PROTOCOL.md` 第 6.2 节，`engram.out` 实测 8.6e-6）。
