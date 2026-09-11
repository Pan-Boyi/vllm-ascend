# dsv41-model — 重启前状态快照

写于宿主机重启前。代码全部已 commit（工作区干净），这份文件记的是**没落进代码的判断**。

## 代码状态

已完成并测通（`tests/ut/attention/` + `tests/ut/models/` 311 + 相关，全绿）：

| 模块 | 状态 |
|---|---|
| `weights_mapper.py` | 完成。96085 个真实张量名 0 unmapped，未匹配抛异常不跳过 |
| `debug_align.py` | 完成。逐层 dump + 强制路由/层输入注入，按全局序列位置寻址 |
| `rope.py` | 完成。逆 RoPE，golden 三层逐位相等 |
| `o_proj.py` | 完成。块对角投影 + e8m0 反量化 |
| `engram_gather.py` | 完成。host 注册 + AIV 直读 + 上游 embedding 的 mixin |
| `attn_selector.py` | 完成。劫持 `_select_dsv4_attn_cls` |
| `platform_patches.py` | 完成。`_JOIN_ROW_PTRS=False`（join 分支 core dump） |
| `attention.py` | **骨架**。`forward_mqa` / `_o_proj` 仍 raise NotImplementedError |
| KV spec | **未开始** |
| attention backend | **未开始** |
| `models/__init__.py` 注册 | **未开始** |

## 只在我脑子里、没写进代码的判断

1. **`dsa_v1.py:2204` 的 `ori_mask_mode=4` 我故意没改。** 它是无条件传的，改它会同时改变 V4.0 行为（不像 `:203` 那条被 `is_a5_bf16_kv_enabled` 守着）。而且它连着更深的一条：`if compress_ratio <= 1` 会跳过整个压缩流，那是 V4.0 语义。**只修 mask mode 不修这条，会从"报错"变成"静默算错"**——layer 20~39 拿不到压缩流而算子不抱怨。这两条要一起改，且需要能端到端验证之后再动。

2. **`supported_compress_ratios` 校验的是 config 值不是算子参数。** 两者不同：config 的 0 合法（layer 0/1），算子的 `cmp_ratio=0` 非法。已写进 `_dsa_swa_only_cmp_ratio` 的 docstring，但如果有人只看 `dsa_attn_kv_plan.py` 会看不到。

3. **legacy plan 的 `supported_compress_ratios=(0,1,4,128)` 是继承 V4.0 的假设，没人查过 `npu_sparse_attn_sharedkv` 的校验器。** 注释里写了"未确认"，别当成已验证。

4. **KV spec 必须按 plan 取**（dtype / bytes-per-token / layout / 块内分区），不能写死 bf16。理由：fp8 那条 plan 的块内是「整块 KV 在前、整块 scale 在后」，连续排布会产 NaN，而那条只写在一个 nightly 测试的注释里。写死的话第二步要重写 KV spec。

5. **rope 怎么进 `sparse_flash_mla` 未确认。** 该算子没有 `rope_head_dim` 参数、q 和 kv 都是 512 宽。推断是"rope 在写 cache 前就做完、算子吃已带 rope 的 512 维"，**但这是推断**。接线时必须给确定答案。

6. **`test_dsa_compress_ratio_guard.py` 会在合并 kernel 分支后变红**，因为我的报错文案说 MQSFM "not bound in this repo yet"（在我分支里目前为真）。**红的原因是措辞过期不是功能坏，不要删测试或改断言让它绿**——那是唯一一个会自动提醒"某句描述过期"的机制。

7. **host 注册测试被 `VLLM_ASCEND_DSV41_ALLOW_HOST_REGISTER=1` 闸住。** 三张卡在注册期间挂过，相关性未证实为因果。放开前先确认卡的问题有结论。

## 下一步
①KV spec（参数化）→ ②backend → ③注册。④⑤⑥⑦ 见协调方的恢复文档。
