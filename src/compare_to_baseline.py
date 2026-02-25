import argparse
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, fftconvolve

EPS = 1e-12

# -----------------------------
# 基础函数：读音频 & 单声道能量平均
# -----------------------------
def read_audio(path: str):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x, sr

def power_mono(x_lr: np.ndarray) -> np.ndarray:
    """
    立体声能量平均 -> 单声道幅度：
    mono = sqrt( 0.5*(L^2 + R^2) )
    """
    if x_lr.ndim != 2:
        raise ValueError("audio must be [N, C]")
    if x_lr.shape[1] == 1:
        return np.abs(x_lr[:, 0])
    p = 0.5 * (x_lr[:, 0] * x_lr[:, 0] + x_lr[:, 1] * x_lr[:, 1])
    return np.sqrt(p + EPS)

def rms_dbfs_from_mono(mono: np.ndarray) -> float:
    r = np.sqrt(np.mean(mono * mono) + EPS)
    return float(20.0 * np.log10(r + EPS))

# -----------------------------
# 对齐：用互相关估计全局延迟（cand - base）
# -----------------------------
def find_delay_by_corr(cand_path, base_path, sr=48000, ds_sr=2000, chunk_sec=25):
    """
    取 base 的中间一段 chunk，与 cand 全长做互相关，估计 delay（cand - base）。
    返回：delay_samples（正数表示 cand 比 base 晚，需要 cand 往前切）
    """
    xb, srb = read_audio(base_path)
    xc, src = read_audio(cand_path)
    if srb != sr or src != sr:
        raise ValueError(f"sample rate mismatch: base={srb}, cand={src}, expected={sr}")

    mb = power_mono(xb)
    mc = power_mono(xc)

    Nbase = len(mb)
    mid = int(0.5 * Nbase)
    half = int(0.5 * chunk_sec * sr)
    s = max(0, mid - half)
    e = min(Nbase, mid + half)
    mb_chunk = mb[s:e].astype(np.float32)

    # 下采样到 ds_sr
    mb_ds = resample_poly(mb_chunk, ds_sr, sr).astype(np.float32)
    mc_ds = resample_poly(mc, ds_sr, sr).astype(np.float32)
    mb_ds -= np.mean(mb_ds)
    mc_ds -= np.mean(mc_ds)

    # corr[k] = sum mc_ds[t] * mb_ds[t-k]
    corr = fftconvolve(mc_ds, mb_ds[::-1], mode="valid")
    k = int(np.argmax(corr))

    base_center_sec = (s + (e - s) // 2) / sr
    cand_center_sec = (k + len(mb_ds) // 2) / ds_sr
    delay_sec = cand_center_sec - base_center_sec
    delay_samples = int(round(delay_sec * sr))
    return delay_samples

# -----------------------------
# 取对齐后的重叠片段（可截取 max_minutes 加速）
# -----------------------------
def get_aligned_overlap(base_path, cand_path, sr=48000, max_minutes=None):
    xb, srb = read_audio(base_path)
    xc, src = read_audio(cand_path)
    if srb != sr or src != sr:
        raise ValueError("sample rate mismatch")

    delay = find_delay_by_corr(cand_path, base_path, sr=sr)
    # cand - base = delay
    base_start = max(0, -delay)
    cand_start = max(0, delay)

    Nbase = xb.shape[0]
    Ncand = xc.shape[0]

    avail = min(Nbase - base_start, Ncand - cand_start)
    if max_minutes is not None:
        avail = min(avail, int(max_minutes * 60 * sr))

    if avail <= 0:
        raise ValueError("no overlap after alignment")

    xb_seg = xb[base_start:base_start + avail, :]
    xc_seg = xc[cand_start:cand_start + avail, :]

    return xb_seg, xc_seg, delay

# -----------------------------
# 计算平均频谱差：ΔdB = base - cand
# 并做 anchored（300-3k 置零）+ 平滑
# -----------------------------
def avg_spectrum_db(mono: np.ndarray, sr: int, n_fft: int, hop: int):
    win = np.hanning(n_fft).astype(np.float32)
    n = len(mono)
    if n < n_fft:
        raise ValueError("segment too short")
    n_frames = 1 + (n - n_fft) // hop
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)

    for i in range(n_frames):
        st = i * hop
        fr = mono[st:st + n_fft] * win
        X = np.fft.rfft(fr)
        P = (X.real * X.real + X.imag * X.imag) + EPS
        acc += 10.0 * np.log10(P)

    acc /= max(n_frames, 1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freqs.astype(np.float32), acc.astype(np.float32)

def smooth_1d(x: np.ndarray, win: int = 31):
    if win <= 1:
        return x.copy()
    w = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xp, w, mode="valid")
    return y.astype(np.float32)

def band_mean(freqs, y_db, f_lo, f_hi):
    m = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(m):
        return float("nan")
    return float(np.mean(y_db[m]))

def compute_metrics(base_seg, cand_seg, sr=48000, n_fft=4096, hop=2048):
    mb = power_mono(base_seg)
    mc = power_mono(cand_seg)

    # 全局增益差（用于诊断）
    rb = np.sqrt(np.mean(mb * mb) + EPS)
    rc = np.sqrt(np.mean(mc * mc) + EPS)
    gain_db = float(20.0 * np.log10((rb + EPS) / (rc + EPS)))

    # 频谱
    freqs, Sb = avg_spectrum_db(mb, sr, n_fft, hop)
    _, Sc = avg_spectrum_db(mc, sr, n_fft, hop)
    delta_raw = (Sb - Sc).astype(np.float32)  # base - cand

    # anchored：把 300-3000Hz 平均置零（更像“形状差”而不是音量差）
    anchor = band_mean(freqs, delta_raw, 300.0, 3000.0)
    delta_anch = (delta_raw - anchor).astype(np.float32)

    # 平滑
    delta_smooth = smooth_1d(delta_anch, win=31)

    # band stats（用 smooth 更稳定）
    bands = [
        ("20-80", 20, 80),
        ("80-200", 80, 200),
        ("200-1k", 200, 1000),
        ("1k-3k", 1000, 3000),
        ("3k-8k", 3000, 8000),
        ("8k-16k", 8000, 16000),
    ]
    stats = {name: band_mean(freqs, delta_smooth, lo, hi) for name, lo, hi in bands}

    # 简单“音乐相似度分数”：忽略 8k+（避免噪声诱导）
    music_err = np.nanmean([
        abs(stats["200-1k"]),
        abs(stats["1k-3k"]),
        abs(stats["3k-8k"]),
    ])
    # 噪声段差异（只做信息，不纳入音乐分数）
    noise_delta = stats["8k-16k"]

    # 时间域 SNR（只是参考：因为滤波后波形不可能完全重合）
    # 先把 cand 按 gain_db 缩放到与 base RMS 一致，再算残差
    g = 10.0 ** (gain_db / 20.0)
    resid = mb - (mc * g)
    snr = float(10.0 * np.log10((np.sum(mb * mb) + EPS) / (np.sum(resid * resid) + EPS)))

    return freqs, delta_raw, delta_anch, delta_smooth, gain_db, anchor, stats, music_err, noise_delta, snr

def frame_rms_dbfs(mono: np.ndarray, sr: int, win_ms=50, hop_ms=25):
    win = int(sr * win_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    win = max(win, 256)
    hop = max(hop, 128)
    n = len(mono)
    if n < win:
        return np.array([0.0]), np.array([rms_dbfs_from_mono(mono)])
    n_frames = 1 + (n - win) // hop
    t = (np.arange(n_frames) * hop) / sr
    y = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        fr = mono[i * hop:i * hop + win]
        y[i] = rms_dbfs_from_mono(fr)
    return t.astype(np.float32), y

# -----------------------------
# 主程序：对比多个 candidates
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidates", required=True, nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=2048)
    ap.add_argument("--max_minutes", type=float, default=8.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseline = args.baseline
    cands = args.candidates

    # 先读基准一段，用于画包络（对每个候选各取相同 overlap）
    results = []

    for cand in cands:
        xb_seg, xc_seg, delay = get_aligned_overlap(
            baseline, cand, sr=args.sr, max_minutes=args.max_minutes
        )
        freqs, d_raw, d_anch, d_smooth, gain_db, anchor, stats, music_err, noise_delta, snr = compute_metrics(
            xb_seg, xc_seg, sr=args.sr, n_fft=args.n_fft, hop=args.hop
        )

        name = os.path.splitext(os.path.basename(cand))[0]
        csv_path = os.path.join(args.outdir, f"diff_{name}.csv")
        np.savetxt(
            csv_path,
            np.column_stack([freqs, d_raw, d_anch, d_smooth]),
            delimiter=",",
            header="freq_hz,delta_raw_db,delta_anchored_db,delta_smooth_db",
            comments=""
        )

        results.append({
            "name": name,
            "path": cand,
            "delay": delay,
            "gain_db": gain_db,
            "anchor_db": anchor,
            "snr": snr,
            "stats": stats,
            "music_err": music_err,
            "noise_delta": noise_delta,
            "freqs": freqs,
            "delta_smooth": d_smooth,
            "xb_seg": xb_seg,
            "xc_seg": xc_seg
        })

    # -------- summary.txt --------
    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Baseline: {baseline}\n")
        f.write(f"Max minutes analyzed: {args.max_minutes}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"[{r['name']}]\n")
            f.write(f"  file: {r['path']}\n")
            f.write(f"  align delay (cand - base): {r['delay']} samples ({r['delay']/args.sr*1000:.2f} ms)\n")
            f.write(f"  rms gain_db (base/cand): {r['gain_db']:.2f} dB\n")
            f.write(f"  anchor(300-3k) removed: {r['anchor_db']:.2f} dB\n")
            f.write(f"  time SNR (ref): {r['snr']:.2f} dB\n")
            f.write("  band delta (dB, baseline - candidate, anchored+smooth):\n")
            for k in ["20-80","80-200","200-1k","1k-3k","3k-8k","8k-16k"]:
                f.write(f"    {k:>7}: {r['stats'][k]:+6.2f}\n")
            f.write(f"  music_err (200-8k abs avg): {r['music_err']:.2f} dB  (越小越像)\n")
            f.write(f"  noise_delta (8k-16k): {r['noise_delta']:+.2f} dB  (越接近0越像录音噪声)\n")
            f.write("\n")

    # -------- delta_overlay.png --------
    plt.figure(figsize=(12, 5))
    for r in results:
        plt.semilogx(r["freqs"], r["delta_smooth"], label=r["name"])
    plt.axhline(0.0, linewidth=1)
    plt.title("Candidate vs Baseline (Delta = base - cand, anchored@300-3k, smooth)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta dB (base - candidate)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "delta_overlay.png"), dpi=160)
    plt.close()

    # -------- env_rms_dbfs.png --------
    # 用第一个候选的 overlap 作为基准包络参考（所有候选都各自与基准 overlap）
    plt.figure(figsize=(12, 6))
    # baseline envelope（取第一个 overlap 的 baseline 段）
    xb0 = power_mono(results[0]["xb_seg"])
    tb, eb = frame_rms_dbfs(xb0, args.sr)
    plt.plot(tb, eb, label="baseline")

    for r in results:
        mc = power_mono(r["xc_seg"])
        tc, ec = frame_rms_dbfs(mc, args.sr)
        plt.plot(tc, ec, label=r["name"], alpha=0.8)

    plt.title("RMS dBFS Envelope (aligned overlap)")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS dBFS")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "env_rms_dbfs.png"), dpi=160)
    plt.close()

    print("Done.")
    print(f"Outputs in: {args.outdir}")
    print(f"  - summary.txt")
    print(f"  - delta_overlay.png")
    print(f"  - env_rms_dbfs.png")
    for r in results:
        print(f"  - diff_{r['name']}.csv")

if __name__ == "__main__":
    main()
