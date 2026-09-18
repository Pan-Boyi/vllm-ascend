#!/usr/bin/env python3
"""KT_OFFLOAD=1(单卡 MoE offload)起飞前的两道闸:卡上显存 + 主机内存(含逐 NUMA 节点)。

只在 serve_v41_bringup.sh 的 KT_OFFLOAD=1 分支里调用;默认(四卡)路径不经过这里。
所有系数【当场从数据算】,不写死:
  · 每专家字节 = 权重文件 header 里主干 40 层路由专家的 weight+scale 之和 / 384
    (09-18 实测 268.95 GiB → 0.7004 GiB/专家;旧经验式的 275.7 含 3 层 MTP 专家 6.72 GiB,单卡不加载)
  · N = --additional-config 里 kt_offload_config 的常驻数(prefix/frequency 取 num_resident_experts;
    explicit 取 resident_experts_path 各层列表长度的最大值)—— 与服务实际读的是同一份,不另设变量
闸 1(HBM):非专家权重 + N × 每专家 × KT_DEVICE_FACTOR ≤ HBM × GPUUTIL − KV/工作区预留
   ⚪ KT_DEVICE_FACTOR 默认 1.0 = 【未校准】:NPU 侧 NZ 格式可能有对齐填充,首次上卡用实测显存校准
闸 2(主机):卸到 host 的专家 (384−N)×每专家 + 加载瞬时 + 进程开销 + 余量 ≤ MemAvailable
   🔴 engram 已铺在 /dev/shm 时,它已经不在 MemAvailable 里 —— 不重复扣;没铺时要加上它
   🔴 kt 按 NUMA 严格绑定:每个节点需要 ≈ (384−N)×每专家/节点数,不够是 OOM,不溢出到别的节点
退出码:0 放行;1 拒绝(并打印账);2 算不出来(配置/数据缺失,按拒绝处理)
"""
import glob, json, os, re, struct, sys

G = 2 ** 30
MODEL = os.environ.get("MODEL", "/workspace/models/DeepSeek-V4.1-Flash")
env = lambda k, d: float(os.environ.get(k, d))


def die(code, msg):
    print(msg); sys.exit(code)


def expert_bytes():
    tot, experts, layers = 0, set(), set()
    for f in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__" or "shared" in k or "dspark" in k or k.startswith("mtp"):
                continue
            m = re.search(r"layers\.(\d+)\..*experts\.(\d+)\.", k)
            if m:
                layers.add(int(m.group(1))); experts.add(int(m.group(2)))
                tot += v["data_offsets"][1] - v["data_offsets"][0]
    if len(experts) != 384 or len(layers) != 40:
        die(2, f"❌ 权重头里数到 {len(layers)} 层 / {len(experts)} 个专家,不是 40/384 —— 系数算不出来,拒绝")
    return tot / 384


def resident_n():
    raw = os.environ.get("KT_ADDITIONAL_CONFIG", "")
    try:
        c = json.loads(raw)["kt_offload_config"]
    except Exception as e:  # noqa: BLE001
        die(2, f"❌ KT_ADDITIONAL_CONFIG 不是含 kt_offload_config 的 JSON({type(e).__name__}): {raw[:120]!r}")
    if c.get("resident_strategy", "prefix") == "explicit":
        p = c.get("resident_experts_path")
        try:
            d = json.load(open(p)); ls = d.get("layers", d)
            vals = ls.values() if isinstance(ls, dict) else ls
            return max(len(v) for v in vals)
        except Exception as e:  # noqa: BLE001
            die(2, f"❌ explicit 策略的 resident_experts_path 读不出来: {p} ({type(e).__name__})")
    n = c.get("num_resident_experts")
    if not isinstance(n, int) or not 0 <= n <= 384:
        die(2, f"❌ num_resident_experts 必须是 0..384 的整数,现在是 {n!r}")
    return n


def meminfo(path="/proc/meminfo"):
    """/proc/meminfo 与 node*/meminfo 两种格式;无 kB 单位的行(HugePages_*)按原值存。"""
    out = {}
    for line in open(path):
        line = re.sub(r"^Node \d+ ", "", line)
        if ":" not in line:
            continue
        k, v = line.split(":", 1); f = v.split()
        if f:
            out[k.strip()] = int(f[0]) * (1024 if len(f) > 1 and f[1] == "kB" else 1)
    return out


def main():
    per = expert_bytes(); N = resident_n(); off = 384 - N
    fac = env("KT_DEVICE_FACTOR", 1.0); other = env("KT_OTHER_WEIGHTS_GIB", 10.4)
    hbm = env("HBM_GIB", 128); util = env("GPUUTIL", 0.9); kv = env("KT_KV_RESERVE_GIB", 25)
    dev = other + N * per / G * fac; budget = hbm * util - kv
    print(f"  ⚪ 每专家(权重头,40 层 weight+scale)= {per/G:.4f} GiB   常驻 N={N}  卸到 host {off} 个")
    print(f"  闸1 HBM: {other:.1f} + {N}×{per/G:.4f}×{fac:g}(设备系数{'，未校准' if fac == 1.0 else ''}) = {dev:.1f} GiB"
          f"  预算 {hbm:g}×{util:g} − KV/工作区 {kv:g} = {budget:.1f} GiB")
    ok = True
    if dev > budget:
        ok = False; print(f"  ❌ 卡上装不下:超出 {dev-budget:.1f} GiB(调小 N 或 KT_KV_RESERVE_GIB)")
    host = off * per / G; trans = env("KT_LOAD_TRANSIENT_GIB", 15); proc = env("KT_PROC_OVERHEAD_GIB", 10)
    margin = env("KT_HOST_MARGIN_GIB", 16)
    mi = meminfo(); avail = mi["MemAvailable"] / G
    staged = sum(os.path.getsize(f) for f in glob.glob("/dev/shm/engram_stage/*") if os.path.isfile(f)) / G
    engram_add = 0.0 if staged > 180 else 188.8
    need = host + trans + proc + margin + engram_add
    print(f"  闸2 主机: 专家 {host:.1f} + 加载瞬时 {trans:g}(推断) + 进程 {proc:g}(待实测) + 余量 {margin:g}"
          f" + engram {engram_add:.1f}{'(已铺在 /dev/shm,已不在 MemAvailable 里,不重复扣)' if engram_add == 0 else '(未铺,要加上)'}"
          f" = {need:.1f} GiB   MemAvailable {avail:.1f} GiB")
    if need > avail:
        ok = False; print(f"  ❌ 主机内存不够:差 {need-avail:.1f} GiB —— 多半有别的大服务同时在跑(四卡服务与单卡线不能并存)")
    nodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
    if nodes:
        per_node = host / len(nodes)
        for nd in nodes:
            m = meminfo(os.path.join(nd, "meminfo"))
            a = (m.get("MemFree", 0) + m.get("FilePages", 0) - m.get("Shmem", 0)) / G
            flag = "✅" if a >= per_node + margin / len(nodes) else "❌"
            if flag == "❌": ok = False
            print(f"  闸2 {os.path.basename(nd)}: 需要 ≈ {per_node:.1f} GiB(kt 严格 NUMA 绑定)  估算可用 {a:.1f} GiB"
                  f"(MemFree+FilePages−Shmem){'' if flag=='✅' else '  ❌ 该节点不够 —— 会 OOM,不会溢出到别的节点'}")
    print("  ✅ KT offload 两道闸通过" if ok else "❌ KT offload 起飞前检查未过,拒绝启动")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001  崩溃 ≠ 拒绝:归为"算不出来",同样不放行
        print(f"❌ KT offload 闸自身出错({type(e).__name__}: {e})—— 算不出来,拒绝启动"); sys.exit(2)
