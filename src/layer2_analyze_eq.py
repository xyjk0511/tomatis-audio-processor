import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, fftconvolve, savgol_filter

EPS = 1e-12

def power_mono(x_lr: np.ndarray) -> np.ndarray:
    p = 0.5 * (x_lr[:, 0] * x_lr[:, 0] + x_lr[:, 1] * x_lr[:, 1])
    return np.sqrt(p + EPS)

def rms_dbfs(mono: np.ndarray) -> float:
    r = np.sqrt(np.mean(mono * mono) + EPS)
    return float(20.0 * np.log10(r + EPS))

def find_delay_by_corr(target_path, base_path, sr=48000, ds_sr=2000, chunk_sec=25.0):
    """估计 delay = target - base（样点）。正数表示 target 更晚开始。"""
    with sf.SoundFile(target_path) as ft, sf.SoundFile(base_path) as fb:
        assert ft.samplerate == sr and fb.samplerate == sr
        assert ft.channels == 2 and fb.channels == 2

        Nbase = fb.frames
        mid = int(0.5 * Nbase)
        half = int(0.5 * chunk_sec * sr)
        s = max(0, mid - half)
        e = min(Nbase, mid + half)

        fb.seek(s)
        xb = fb.read(e - s, dtype="float32", always_2d=True)
        mb = power_mono(xb)
        mb_ds = resample_poly(mb, ds_sr, sr).astype(np.float32)
        mb_ds -= np.mean(mb_ds)

        ft.seek(0)
        mt_chunks = []
        block = sr * 30
        while True:
            x = ft.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            mt_chunks.append(power_mono(x))
        mt = np.concatenate(mt_chunks).astype(np.float32)
        mt_ds = resample_poly(mt, ds_sr, sr).astype(np.float32)
        mt_ds -= np.mean(mt_ds)

    corr = fftconvolve(mt_ds, mb_ds[::-1], mode="valid")
    k = int(np.argmax(corr))
    base_center_sec = (s + (e - s) // 2) / sr
    targ_center_sec = (k + len(mb_ds) // 2) / ds_sr
    delay_sec = targ_center_sec - base_center_sec
    return int(round(delay_sec * sr))

def stft_logpower_median(x_lr: np.ndarray, sr: int, n_fft: int, hop: int, music_dbfs: float):
    """
    返回：freqs, median_logP (dB), used_frames_count
    只使用 base_level > music_dbfs 的帧，避免静音/噪声地板干扰。
    """
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # 逐帧计算 log power，收集后取 median（抗少量爆点/异常强）
    n_frames = 1 + (len(x_lr) - n_fft) // hop
    if n_frames <= 10:
        raise ValueError("片段太短，无法做稳定频谱统计。")

    logs = []
    used = 0
    for i in range(n_frames):
        st = i * hop
        fr = x_lr[st:st + n_fft, :]
        mono = power_mono(fr)
        lv = rms_dbfs(mono)
        if lv <= music_dbfs:
            continue
        mono_w = mono * win
        X = np.fft.rfft(mono_w)
        P = (X.real * X.real + X.imag * X.imag).astype(np.float32)
        logP = 10.0 * np.log10(P + EPS)
        logs.append(logP)
        used += 1

    if used < 50:
        raise ValueError(f"可用音乐帧太少（{used} 帧）。把 --music_dbfs 调低一点（例如 -70）。")

    logs = np.stack(logs, axis=0).astype(np.float32)  # [T, F]
    med = np.median(logs, axis=0).astype(np.float32)
    return freqs, med, used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基准（录音参照）")
    ap.add_argument("--target", required=True, help="要匹配的目标（D_MNF_matched_v2）")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--max_minutes", type=float, default=6.0, help="用于统计的最长时长（分钟）")
    ap.add_argument("--n_fft", type=int, default=8192)
    ap.add_argument("--hop", type=int, default=4096)
    ap.add_argument("--music_dbfs", type=float, default=-65.0)
    ap.add_argument("--anchor_lo", type=float, default=300.0)
    ap.add_argument("--anchor_hi", type=float, default=3000.0)
    ap.add_argument("--clamp_db", type=float, default=12.0, help="限制 EQ 幅度，避免过拟合/失真")
    ap.add_argument("--smooth_bins", type=int, default=71, help="频率方向平滑窗口（奇数更好）")
    ap.add_argument("--out_csv", default="layer2_eq_curve.csv")
    ap.add_argument("--out_png", default="layer2_eq_curve.png")
    args = ap.parse_args()

    sr = args.sr
    delay = find_delay_by_corr(args.target, args.base, sr=sr)
    print(f"[ALIGN] delay (target - base): {delay} samples ({delay/sr*1000:.2f} ms)")

    with sf.SoundFile(args.base) as fb, sf.SoundFile(args.target) as ft:
        assert fb.samplerate == sr and ft.samplerate == sr
        assert fb.channels == 2 and ft.channels == 2

        base_start = max(0, -delay)
        targ_start = max(0,  delay)

        max_len = int(args.max_minutes * 60 * sr)
        avail = min(fb.frames - base_start, ft.frames - targ_start, max_len)
        if avail <= args.n_fft:
            raise ValueError("对齐后可用重叠太短，无法统计。")

        fb.seek(base_start)
        ft.seek(targ_start)
        xb = fb.read(avail, dtype="float32", always_2d=True)
        xt = ft.read(avail, dtype="float32", always_2d=True)

    # 统计两者的 median log-power spectrum
    freqs, med_b, used_b = stft_logpower_median(xb, sr, args.n_fft, args.hop, args.music_dbfs)
    _,     med_t, used_t = stft_logpower_median(xt, sr, args.n_fft, args.hop, args.music_dbfs)
    used = min(used_b, used_t)
    print(f"[STATS] used music frames (base/target): {used_b}/{used_t}")

    # 差值：base - target（需要给 target 补多少 dB）
    delta = (med_b - med_t).astype(np.float32)

    # 去掉“整体增益偏移”：锚定中频段为 0（避免层 2 把整体响度也拟合进去）
    anchor_mask = (freqs >= args.anchor_lo) & (freqs <= args.anchor_hi)
    anchor = float(np.median(delta[anchor_mask]))
    delta0 = (delta - anchor).astype(np.float32)

    # 限幅，防止极端频点导致失真或过拟合
    clamp = float(args.clamp_db)
    delta0 = np.clip(delta0, -clamp, +clamp)

    # 平滑（频率方向）
    w = int(args.smooth_bins)
    if w % 2 == 0:
        w += 1
    w = max(11, w)
    if w >= len(delta0):
        w = len(delta0) - 1 if (len(delta0) - 1) % 2 == 1 else len(delta0) - 2
    delta_s = savgol_filter(delta0, window_length=w, polyorder=3).astype(np.float32)

    # 只输出可听频段（但保存全频也可以）
    out = np.stack([freqs, delta0, delta_s], axis=1)
    np.savetxt(args.out_csv, out, delimiter=",", header="freq_hz,delta_db_raw,delta_db_smooth", comments="")
    print(f"[SAVED] {args.out_csv}")
    print(f"[INFO] anchor(median {args.anchor_lo}-{args.anchor_hi}Hz) = {anchor:+.2f} dB (removed)")

    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, delta0, label="raw (anchored, clamped)")
    plt.plot(freqs, delta_s, label="smooth")
    plt.xscale("log")
    plt.xlim(20, sr/2)
    plt.ylim(-clamp-1, clamp+1)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta (dB)  [base - target]")
    plt.title("Layer2 EQ Curve (Static)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[SAVED] {args.out_png}")

if __name__ == "__main__":
    main()
