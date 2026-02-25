# plot_diff_spectrum.py
# 作用：读取 diff_spectrum.csv 并画频谱差异曲线 + 频段均值统计
# 约定：delta_db = base - candidate（正：基准更响/更亮）

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def band_mean(freq, delta, f_lo, f_hi):
    m = (freq >= f_lo) & (freq < f_hi) & np.isfinite(delta)
    if not np.any(m):
        return float("nan")
    return float(np.mean(delta[m]))

def smooth_logfreq(freq, y, win=31):
    """
    在 log10(freq) 上做一个简单的滑动平均（等价于“按倍频程尺度更平滑”）
    实现：把点按 logf 排序，然后对 y 做 moving average。
    """
    freq = np.asarray(freq)
    y = np.asarray(y)
    m = np.isfinite(freq) & np.isfinite(y) & (freq > 0)
    freq = freq[m]
    y = y[m]
    order = np.argsort(freq)
    freq = freq[order]
    y = y[order]

    # moving average
    win = int(win)
    win = max(3, win | 1)  # 强制奇数且 >=3
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    ys = np.convolve(ypad, kernel, mode="valid")
    return freq, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="diff_spectrum.csv 路径")
    ap.add_argument("-o", "--out_png", default=None, help="输出 png 路径（默认同名 .png）")
    ap.add_argument("--title", default="Diff Spectrum (base - candidate)", help="图标题")
    ap.add_argument("--smooth_win", type=int, default=31, help="平滑窗口长度(点数)，奇数更好")
    ap.add_argument("--xlim", type=float, nargs=2, default=[20, 20000], help="频率范围 Hz")
    args = ap.parse_args()

    in_path = args.input
    if args.out_png is None:
        base, _ = os.path.splitext(in_path)
        out_png = base + ".png"
    else:
        out_png = args.out_png

    df = pd.read_csv(in_path)

    # 兼容不同列名
    f_col = pick_col(df, ["freq_hz", "frequency_hz", "freq", "frequency", "Hz", "hz"])
    d_col = pick_col(df, ["delta_db_base_minus_cand", "delta_db", "diff_db", "delta", "dB", "db"])

    if f_col is None or d_col is None:
        raise ValueError(
            f"找不到列名。需要频率列(如 freq_hz)和差值列(如 delta_db_base_minus_cand)。\n"
            f"当前列名: {list(df.columns)}"
        )

    freq = df[f_col].to_numpy(dtype=np.float64)
    delta = df[d_col].to_numpy(dtype=np.float64)

    # 频段均值（按你的常用分段）
    bands = [
        (200, 1000),
        (1000, 3000),
        (3000, 8000),
        (8000, 16000),
    ]
    print("Band mean (dB), delta = base - candidate:")
    for lo, hi in bands:
        m = band_mean(freq, delta, lo, hi)
        print(f"  {lo:>5}-{hi:<5} Hz : {m:+.2f} dB")

    # 平滑曲线
    fs, ds = smooth_logfreq(freq, delta, win=args.smooth_win)

    # 画图
    plt.figure(figsize=(14, 7))
    plt.plot(freq, delta, linewidth=1.2, label="raw")
    plt.plot(fs, ds, linewidth=2.0, label=f"smooth (win={args.smooth_win})")
    plt.axhline(0.0, linewidth=1.0)

    plt.xscale("log")
    plt.xlim(args.xlim[0], args.xlim[1])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta dB (base - candidate)")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"\nSaved: {out_png}")

if __name__ == "__main__":
    main()
