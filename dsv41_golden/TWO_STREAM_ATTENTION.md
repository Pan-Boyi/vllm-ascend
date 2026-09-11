# ori 流 × cmp 流 —— attention 里的合并语义(权威定义)

给 `dsv41-kernel` 写 `sparse_flash_mla` 两流参考用的。**独立于任何算子实现**,
全部来自 `ref/`(`/workspace/models/DeepSeek-V4.1-Flash/.../inference/` 的逐字副本,
只把 `kernel.py` 换成 CPU 版),所以这里的数值定义就是官方定义。

> 🔴 **为什么单独成文**:kernel 刻意不去读 CANN kernel 源自己推参考 ——
> **从被测对象推出来的参考,和被测对象共享同一个错误,会给出一个「看起来通过」的 `ok`。**
> 所以它需要一份独立来源。这份就是。

---

## 1. 两条流怎么合:一次 concat,一次 softmax

`ref/model.py:776-780`:

```python
kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
if self.compress_ratio:
    compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))   # 776 ← offset
    kv        = torch.cat([kv, compress_kv], dim=1)             # 777
    topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)   # 778
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)   # 780 ← 唯一一次调用
```

**整个 forward 只有一个 `sparse_attn` 调用,代码里任何地方都没有 lse 合并。**
ori 行和 cmp 行拼成**一条序列**,`topk_idxs` 索引的是**拼接后的轴**。

⇒ **不是**「各自算完再按 lse 合并」。猜成那样会得到一个**有限的、形状正确的、错 0.65 的**答案
(kernel 实测),而两者都过不了任何形状检查 —— 这正是猜出来的参考的危险形态。

## 2. 🔴 cmp 索引带偏移 = 窗口长度

`_compress_kv` 的第四个实参就是 `kv.size(1)`(第 776 行)。这是我 dump 里
`indexer_kv_offset` 记的量。

- **prefill 窗口长 `seqlen`,decode 窗口长 `window_size`** ⇒ **同一个压缩位置在两条路径上编号不同。**
- 不减掉它,索引比对会得到 **100% 假不匹配**。
- **`-1` 空槽不参与平移。**

## 3. sink:在**合并后的**分母里,只加一次

`ref/kernel_cpu.py:469-471`:

```python
mx    = scores.amax(dim=-1).clamp_min(-1e30)      # 对【两条流全部 slot】取一个 max
p     = torch.exp(scores - mx.unsqueeze(-1))
denom = p.sum(dim=-1) + torch.exp(sink.view(1,1,h) - mx)   # sink 只加这一次
```

- **一个 max、一个 sum、sink 加一次。没有 per-stream sink,没有 per-stream 归一化。**
- sink 是**每 head 一个可学习 logit,只进分母,没有 value 向量**
  (`kernel_cpu.py:434-435`)⇒ 分子不受影响;它的作用是让 head 能「attend 到空」。

## 4. 🔴 running max 的地板是有限的 `-1e30`,不是 `-inf`

整行 `topk_idxs` 全 `-1` 时:分子 `exp(-inf+1e30)=0`、分母 `exp(sink+1e30)=+inf`
⇒ **输出全 0 而不是 NaN**。这是训练 kernel 的约定,tilelang 那版刻意复现。

⚠️ **27 token 标准输入在 layer 2 上第 0 行就是整行 `-1`** —— 不用构造就会撞上
(`golden_dumps/full40/layer_02` 实测每行 `-1` 个数
`[13,12,12,11,11,10,...,1,1,0,0]`,第 0 行 13/13 全是 `-1`)。

⚠️ **这条语义必须是「可观察的」,不能被参考里的「贴心掩码」吃掉。**
kernel 第一版在 softmax 后加了 `p = torch.where(valid, p, 0)`(`kernel_cpu.py:470` 没有),
输出是对的,**但 `-1e30` 和 `-inf` 两个地板给出完全相同的结果** ⇒
**一个观察不到这条语义的参考,也检测不出算子在这里选错了。**
去掉之后「换 `-inf` 地板同一行变 NaN」这条反例才能失败。

`-1` 空槽:KV 行读作 0、score 读作 `-inf`(`kernel_cpu.py:427`)。

## 5. `kv` 同时当 K 和 V(latent attention)

`kernel_cpu.py:422`:`kv` [b, n, d] 被**同时**用作 K 和 V,不是两个张量。
可观察的反例:**把 `kv` 乘 3,输出不是原来的 3 倍**(独立的 V 会正好 3 倍)。

## 6. 🔴 两条流的量化方式不同

| 流 | 量化 | 行号 |
|---|---|---|
| 窗口 KV | **fp8 e4m3**,block `fp8_block_size`(=32),`scale_fmt`/`scale_dtype` 来自 args | `model.py:707` |
| 压缩 latent | **fp4 e2m1,group 16,e4m3 scales** | `model.py:760` |

两处都是 `inplace=True`,语义是**把 `dequant(quant(x))` 写回 `x`**
(`kernel_cpu.py:192-193`)⇒ **拼接后的 `kv` 是 bf16,但两半带着各自的量化噪声。**

⚠️ **一个把两半按同样方式量化的参考,会差出一个看起来像算子 bug 的量。**

## 7. rope:三处细节 + 一次输出上的逆旋转

- 窗口 K:`model.py:706` 先 rope、`707` 再量化 ⇒ **量化覆盖 post-RoPE 的整个向量,含 rope 尾**
  (第 700-702 行 docstring 明写)。
- q:`model.py:772`。
- 压缩 latent:`model.py:758`,**位置是 `j * ratio`** —— 一个 latent 代表它那组的**第一个** token
  (第 752-757 行),**不是它自己的下标**。
- theta 分两档:有压缩的层用 `compress_rope_theta` + YaRN;纯滑窗层**关掉 YaRN、用基础
  `rope_theta`**(`model.py:683-687`)。**同一层内 q / 窗口 K / latent 共用同一份 `freqs_cis`**,
  差别在层与层之间。

🔴 **输出上有一次逆 rope,别漏:**

```python
o = sparse_attn(...)                              # 780
apply_rotary_emb(o[..., -rd:], freqs_cis, True)   # 781  ← inverse=True
```

`sparse_attn` 返回后立刻对输出的 rope 尾做**逆旋转**。若算子契约是「返回 attend 结果」,
这一步是调用方的活;**一个漏了它的参考,后面全错**。

⇒ dump 里两个点专为此留:**`attn.o_raw`(逆 rope 之前)** 与 **`attn.o_unroped`(之后)**。
kernel 的参考返回 `sparse_attn` 原始输出 ⇒ **对 `attn.o_raw` 那一侧**。

## 8. 带这些 key 的 dump

| dump | `attn.o_raw` | `attn.o_unroped` | `indexer_kv_offset` | `indexer_score` |
|---|---|---|---|---|
| `smoke3` | ✗ | ✗ | ✗ | ✗ |
| `full40_v2` | ✗ | ✗ | ✓ | ✗ |
| **`len256`**(T=256,3 层) | **✓** | **✓** | **✓** | **✓** |

⇒ **`len256` / `len256_iso` 是目前唯一带全四个 key 的**(`attn.o_*` 那两个 tap 比
`smoke3`/`full40_v2` 晚加)。要 27 token 版的话重跑一次 `--layers 3` 即可(约 20 秒,不占卡),
见协议第 9 节。

## 9. 对 ratio=2 参考的建议:ratio=1 的部分一行都不用改

softmax、sink、`-1e30` 地板**本来就作用在合并后的序列上** ⇒
ratio=1 已经做对的那些,在 ratio=2 下不变。**真正的新代码只有「压缩流怎么构造」** ——
多做一份 fp4-group16-e4m3 的 latent、concat、给 cmp 索引加窗口长度偏移。
