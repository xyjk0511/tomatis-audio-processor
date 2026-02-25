import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, fftconvolve, get_window

EPS = 1e-12

def power_mono(x_lr: np.ndarray) -> np.ndarray:
    # x_lr: [N,2]
    p = 0.5 * (x_lr[:, 0] * x_lr[:, 0] + x_lr[:, 1] * x_lr[:, 1])
    return np.sqrt(p + EPS)

def rms_dbfs_vec(x: np.ndarray) -> float:
    r = np.sqrt(np.mean(x * x) + EPS)
    return float(20.0 * np.log10(r + EPS))

def read_mono_ds(path: str, sr: int, ds_sr: int, block_sec: int = 30) -> np.ndarray:
    """流式读音频 -> mono(power) -> 下采样到 ds_sr，返回 float32 1D"""
    with sf.SoundFile(path) as f:
        assert f.samplerate == sr, f"{path}: sr={f.samplerate} != {sr}"
        assert f.channels == 2, f"{path}: channels={f.channels} != 2"
        blocks = []
        block = sr * block_sec
        while True:
            x = f.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            m = power_mono(x)
            # resample_poly(up, down): from sr -> ds_sr
            m_ds = resample_poly(m, ds_sr, sr).astype(np.float32)
            blocks.append(m_ds)
        return np.concatenate(blocks) if blocks else np.zeros((0,), np.float32)

def find_delay_by_corr(base_path: str, cand_path: str, sr=48000, ds_sr=2000, chunk_sec=25) -> int:
    """
    估计 cand 相对 base 的延迟（单位：samples at sr）
    思路：取 base 中间 chunk，下采样；与 cand 全长下采样做互相关，找最大相关位置。
    返回：delay_samples = cand_time - base_time（cand 比 base 晚为正）
    """
    # base chunk
    with sf.SoundFile(base_path) as fb:
        Nbase = fb.frames
        mid = int(0.5 * Nbase)
        half = int(0.5 * chunk_sec * sr)
        s = max(0, mid - half)
        e = min(Nbase, mid + half)
        fb.seek(s)
        xb = fb.read(e - s, dtype="float32", always_2d=True)
        mb = power_mono(xb)
    mb_ds = resample_poly(mb, ds_sr, sr).astype(np.float32)
    mb_ds -= float(np.mean(mb_ds))

    # cand full
    mc_ds = read_mono_ds(cand_path, sr=sr, ds_sr=ds_sr)
    mc_ds -= float(np.mean(mc_ds))

    # corr[k] = sum cand[t] * base[t-k]
    rb = mb_ds[::-1]
    corr = fftconvolve(mc_ds, rb, mode="valid")
    k = int(np.argmax(corr))  # base_chunk 对齐到 cand 的位置（下采样坐标）

    base_center_sec = (s + (e - s) // 2) / sr
    cand_center_sec = (k + len(mb_ds) // 2) / ds_sr
    delay_sec = cand_center_sec - base_center_sec
    delay_samples = int(round(delay_sec * sr))
    return delay_samples

def read_aligned_overlap(base_path: str, cand_path: str, delay_samples: int, sr=48000, max_sec=None):
    """
    按 delay 对齐，读出两者重叠区间（float32 stereo）
    delay_samples = cand - base
    """
    with sf.SoundFile(base_path) as fb, sf.SoundFile(cand_path) as fc:
        Nb, Nc = fb.frames, fc.frames
        base_start = max(0, -delay_samples)
        cand_start = max(0, delay_samples)
        avail = min(Nb - base_start, Nc - cand_start)
        if max_sec is not None:
            avail = min(avail, int(max_sec * sr))
        if avail <= 0:
            raise ValueError("No overlap after alignment.")
        fb.seek(base_start)
        fc.seek(cand_start)
        xb = fb.read(avail, dtype="float32", always_2d=True)
        xc = fc.read(avail, dtype="float32", always_2d=True)
    return xb, xc

def rms_envelope_dbfs(x_lr: np.ndarray, sr: int, win_sec: float = 0.05, hop_sec: float = 0.01) -> np.ndarray:
    """短窗 RMS(dBFS) 包络：默认 50ms 窗，10ms hop"""
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if win < 32: win = 32
    if hop < 1: hop = 1
    m = power_mono(x_lr)
    n = len(m)
    out = []
    for s in range(0, n - win + 1, hop):
        out.append(rms_dbfs_vec(m[s:s+win]))
    return np.array(out, np.float32), hop / sr

def avg_spectrum_db(x_lr: np.ndarray, sr: int, n_fft: int = 8192, hop: int = 4096):
    """
    Welch 风格平均功率谱（mono power），返回 freqs, spec_db
    """
    m = power_mono(x_lr).astype(np.float32)
    win = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    win_pow = float(np.sum(win * win) + EPS)

    n = len(m)
    acc = None
    count = 0
    for s in range(0, n - n_fft + 1, hop):
        frame = m[s:s+n_fft] * win
        X = np.fft.rfft(frame)
        P = (X.real * X.real + X.imag * X.imag).astype(np.float64) / win_pow
        acc = P if acc is None else (acc + P)
        count += 1

    if acc is None:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        return freqs, np.full_like(freqs, -120.0, dtype=np.float32)

    acc /= max(count, 1)
    spec_db = 10.0 * np.log10(acc + EPS)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freqs, spec_db.astype(np.float32)

def smooth_ma(y: np.ndarray, win: int = 31) -> np.ndarray:
    if win <= 1:
        return y.copy()
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(yp, k, mode="valid").astype(np.float32)

def band_mean(freqs, y, f1, f2):
    m = (freqs >= f1) & (freqs < f2)
    if not np.any(m):
        return float("nan")
    return float(np.mean(y[m]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline FLAC (Tomatis_D_30m_declick 或你认定的基准)")
    ap.add_argument("--cand", required=True, nargs="+", help="candidate FLACs (你的版本/Matlab版本等，支持多个)")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--ds_sr", type=int, default=2000)
    ap.add_argument("--plot_sec", type=float, default=500.0, help="包络图展示前多少秒")
    ap.add_argument("--max_sec_spec", type=float, default=600.0, help="算频谱用的最大秒数（越大越稳但越慢）")
    ap.add_argument("--n_fft", type=int, default=8192)
    ap.add_argument("--hop", type=int, default=4096)
    ap.add_argument("--smooth_win", type=int, default=31)
    ap.add_argument("--out_prefix", default="cmp", help="输出文件前缀")
    args = ap.parse_args()

    base = args.base
    cands = args.cand

    # 先读 baseline 一次（用于频谱/包络时域）
    results = []

    # 用于画包络：读对齐后重叠的前 plot_sec 秒
    plt.figure(1)
    # baseline 包络（对齐时用第一个 cand 的 delay 来切 overlap，避免每条线时间轴不一致）
    delay0 = find_delay_by_corr(base, cands[0], sr=args.sr, ds_sr=args.ds_sr)
    xb0, xc0 = read_aligned_overlap(base, cands[0], delay0, sr=args.sr, max_sec=args.plot_sec)
    env_b, dt = rms_envelope_dbfs(xb0, args.sr)
    t = np.arange(len(env_b)) * dt
    plt.plot(t, env_b, label="baseline")

    for cand in cands:
        try:
            delay = find_delay_by_corr(base, cand, sr=args.sr, ds_sr=args.ds_sr)
            xb, xc = read_aligned_overlap(base, cand, delay, sr=args.sr, max_sec=args.plot_sec)
            env_c, _ = rms_envelope_dbfs(xc, args.sr)
            plt.plot(t[:len(env_c)], env_c, label=cand.split("\\")[-1].split("/")[-1])

            # 频谱对比（用 max_sec_spec）
            xb_s, xc_s = read_aligned_overlap(base, cand, delay, sr=args.sr, max_sec=args.max_sec_spec)

            freqs, sb = avg_spectrum_db(xb_s, args.sr, n_fft=args.n_fft, hop=args.hop)
            _, sc = avg_spectrum_db(xc_s, args.sr, n_fft=args.n_fft, hop=args.hop)

            delta = (sb - sc).astype(np.float32)  # base - cand
            # anchor@300-3k：去掉整体偏移
            anchor = band_mean(freqs, delta, 300, 3000)
            delta_a = (delta - anchor).astype(np.float32)
            delta_s = smooth_ma(delta_a, win=args.smooth_win)

            # 统计
            b_200_1k = band_mean(freqs, delta_a, 200, 1000)
            b_1k_3k  = band_mean(freqs, delta_a, 1000, 3000)
            b_3k_8k  = band_mean(freqs, delta_a, 3000, 8000)
            b_8k_16k = band_mean(freqs, delta_a, 8000, 16000)

            results.append({
                "cand": cand,
                "delay_samples": delay,
                "delay_ms": delay / args.sr * 1000.0,
                "anchor_300_3k_db": anchor,
                "band_200_1k_db": b_200_1k,
                "band_1k_3k_db": b_1k_3k,
                "band_3k_8k_db": b_3k_8k,
                "band_8k_16k_db": b_8k_16k,
            })

            # 保存每个 cand 的 delta csv
            csv_path = f"{args.out_prefix}_diff_{cand.split('\\')[-1].split('/')[-1]}.csv"
            np.savetxt(csv_path,
                       np.stack([freqs, delta_a, delta_s], axis=1),
                       delimiter=",",
                       header="freq_hz,delta_db_anchored,delta_db_smooth",
                       comments="")
            print(f"[WROTE] {csv_path}")

            # 画频谱差（叠加）
            plt.figure(2)
            plt.semilogx(freqs[1:], delta_s[1:], label=cand.split("\\")[-1].split("/")[-1])
            
        except Exception as e:
            print(f"Error processing {cand}: {e}")

    # 包络图输出
    plt.figure(1)
    plt.title("RMS dBFS Envelope (aligned overlap)")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS dBFS")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right")
    env_png = f"{args.out_prefix}_env_rms_dbfs.png"
    plt.savefig(env_png, dpi=160)
    print(f"[WROTE] {env_png}")

    # 频谱差叠加图输出
    plt.figure(2)
    plt.title("Candidate vs Baseline (Delta = base - cand, anchored@300-3k, smooth)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta dB (base - candidate)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.axhline(0.0, linewidth=1)
    plt.legend(loc="best")
    delta_png = f"{args.out_prefix}_delta_overlay.png"
    plt.savefig(delta_png, dpi=160)
    print(f"[WROTE] {delta_png}")

    # summary.txt
    lines = []
    lines.append("=== Comparison Summary ===")
    lines.append(f"base: {base}")
    for r in results:
        lines.append("")
        lines.append(f"cand: {r['cand']}")
        lines.append(f"  delay: {r['delay_samples']} samples ({r['delay_ms']:.2f} ms)")
        lines.append(f"  anchor@300-3k (mean delta): {r['anchor_300_3k_db']:+.2f} dB")
        lines.append(f"  band 200-1k : {r['band_200_1k_db']:+.2f} dB")
        lines.append(f"  band 1k-3k  : {r['band_1k_3k_db']:+.2f} dB")
        lines.append(f"  band 3k-8k  : {r['band_3k_8k_db']:+.2f} dB")
        lines.append(f"  band 8k-16k : {r['band_8k_16k_db']:+.2f} dB")

    summary_path = f"{args.out_prefix}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WROTE] {summary_path}")

if __name__ == "__main__":
    main()
