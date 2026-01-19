import argparse
import numpy as np
import soundfile as sf

EPS = 1e-12

def mad_sigma(x: np.ndarray) -> float:
    """鲁棒尺度估计：MAD -> sigma"""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    return float(mad / 0.6745)

def merge_runs(mask: np.ndarray, gap: int = 0) -> np.ndarray:
    """
    把 True mask 合并成区间 [start,end) 列表；gap 允许把很近的区间合并
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    # 找断点
    cut = np.where(np.diff(idx) > (1 + gap))[0]
    starts = np.r_[idx[0], idx[cut + 1]]
    ends   = np.r_[idx[cut] + 1, idx[-1] + 1]
    return np.stack([starts, ends], axis=1).astype(np.int64)

def inpaint_linear(x: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """
    对每个区间做线性插值修复。x shape [N, C]
    """
    y = x.copy()
    N, C = y.shape
    for s, e in segs:
        s0 = max(0, s - 1)
        e0 = min(N - 1, e)
        if s0 >= e0:
            continue
        for c in range(C):
            left = y[s0, c]
            right = y[e0, c]
            L = e0 - s0
            # 在 (s0, e0) 之间插值；注意把 [s,e) 覆盖掉
            t = np.linspace(0.0, 1.0, L + 1, dtype=np.float32)
            interp = (1 - t) * left + t * right
            # 写回 s..e-1
            y[s:e, c] = interp[(s - s0):(e - s0),]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--k", type=float, default=12.0, help="阈值系数：越大越保守（建议 10-18）")
    ap.add_argument("--pad_ms", type=float, default=1.5, help="每个爆点左右扩展窗口（ms）")
    ap.add_argument("--merge_gap_ms", type=float, default=0.5, help="把相近爆点合并（ms）")
    ap.add_argument("--max_fix_ms", type=float, default=8.0, help="单次修复最大长度（ms），更长就跳过以免误伤")
    ap.add_argument("--report_csv", default=None, help="可选：输出爆点区间列表 CSV")
    args = ap.parse_args()

    x, sr = sf.read(args.input, dtype="float32", always_2d=True)
    N, C = x.shape
    print(f"[LOAD] sr={sr}, shape={x.shape}")

    # 1) 用差分检测冲击（对任一声道）
    dx = np.diff(x, axis=0)  # [N-1,C]
    # 用所有声道差分的最大值做检测统计
    dmax = np.max(np.abs(dx), axis=1)  # [N-1]
    sigma = mad_sigma(dmax)
    thr = args.k * sigma
    hit = dmax > thr  # 命中点在差分索引上

    print(f"[DETECT] MAD-sigma={sigma:.6g}, thr={thr:.6g}, hits={int(hit.sum())}")

    if hit.sum() == 0:
        print("[DONE] 未检测到明显爆点，直接拷贝输出。")
        sf.write(args.output, x, sr, format="FLAC", subtype="PCM_24")
        return

    pad = int(round(args.pad_ms * sr / 1000.0))
    gap = int(round(args.merge_gap_ms * sr / 1000.0))
    max_fix = int(round(args.max_fix_ms * sr / 1000.0))

    # 2) 把 hit 扩展到样点 mask（注意 hit 对应差分位置 i -> 影响 i 和 i+1）
    mask = np.zeros(N, dtype=bool)
    hit_idx = np.flatnonzero(hit)
    for i in hit_idx:
        s = max(0, i - pad)
        e = min(N, i + 1 + pad)
        mask[s:e] = True

    segs = merge_runs(mask, gap=gap)
    # 3) 过滤太长区间（避免把真实音乐突变误判成噪声）
    keep = (segs[:, 1] - segs[:, 0]) <= max_fix
    segs2 = segs[keep]
    print(f"[SEGS] raw={len(segs)}, kept={len(segs2)} (drop long={len(segs)-len(segs2)})")

    # 4) 插值修复
    y = inpaint_linear(x, segs2)

    # 5) 可选报告
    if args.report_csv:
        import csv
        with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_sample", "end_sample", "start_sec", "end_sec", "len_samples"])
            for s, e in segs2:
                w.writerow([int(s), int(e), s/sr, e/sr, int(e-s)])
        print(f"[REPORT] wrote {args.report_csv}")

    sf.write(args.output, y, sr, format="FLAC", subtype="PCM_24")
    print(f"[SAVE] {args.output}")

if __name__ == "__main__":
    main()
