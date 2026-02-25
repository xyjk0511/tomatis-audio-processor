import argparse
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve
import matplotlib.pyplot as plt

EPS = 1e-12

def power_mono_lr(x_lr: np.ndarray) -> np.ndarray:
    """
    立体声 -> 单通道（能量平均得到“幅度”）
    mono = sqrt( (L^2 + R^2)/2 )
    """
    p = 0.5 * (x_lr[:, 0] * x_lr[:, 0] + x_lr[:, 1] * x_lr[:, 1])
    return np.sqrt(p + EPS).astype(np.float32)

def rms_dbfs_mono(mono: np.ndarray) -> float:
    r = np.sqrt(np.mean(mono * mono) + EPS)
    return float(20.0 * np.log10(r + EPS))

def read_segment(path: str, start: int, length: int, dtype="float32") -> np.ndarray:
    with sf.SoundFile(path) as f:
        f.seek(start)
        x = f.read(length, dtype=dtype, always_2d=True)
    return x

def find_delay_by_corr(orig_path, base_path, sr=48000, ds_sr=2000, chunk_sec=25, base_chunk_pos="mid"):
    """
    用互相关估计“orig 相对 base”的延迟（样点）。
    返回：delay_samples = (orig - base)，>0 表示 orig 比 base 晚（需要从 orig 往后切）。
    """
    with sf.SoundFile(orig_path) as fo, sf.SoundFile(base_path) as fb:
        assert fo.samplerate == sr and fb.samplerate == sr, "采样率不一致"
        assert fo.channels == 2 and fb.channels == 2, "只支持双声道"
        Norig = fo.frames
        Nbase = fb.frames

        # 取 base 的一个 chunk（默认中间），避免开头/结尾的静音/杂音影响
        half = int(0.5 * chunk_sec * sr)
        if base_chunk_pos == "mid":
            mid = int(0.5 * Nbase)
            s = max(0, mid - half)
        elif base_chunk_pos == "start":
            s = 0
        else:  # "end"
            s = max(0, Nbase - 2 * half)
        e = min(Nbase, s + 2 * half)

        fb.seek(s)
        xb = fb.read(e - s, dtype="float32", always_2d=True)
        mb = power_mono_lr(xb)
        mb_ds = resample_poly(mb, ds_sr, sr).astype(np.float32)
        mb_ds -= np.mean(mb_ds)

        # 原始读全长（mono）后下采样（30min 也还行；不想全读可改成分块拼接）
        fo.seek(0)
        blocks = []
        block = sr * 30  # 30s/block
        while True:
            x = fo.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            blocks.append(power_mono_lr(x))
        mo = np.concatenate(blocks).astype(np.float32)
        mo_ds = resample_poly(mo, ds_sr, sr).astype(np.float32)
        mo_ds -= np.mean(mo_ds)

    rb = mb_ds[::-1]
    corr = fftconvolve(mo_ds, rb, mode="valid")
    k = int(np.argmax(corr))  # base_chunk 在 orig_ds 里的起点（下采样坐标）

    # 用 chunk 的中心点做“时间对齐”
    base_center_sec = (s + (e - s) // 2) / sr
    orig_center_sec = (k + len(mb_ds) // 2) / ds_sr
    delay_sec = orig_center_sec - base_center_sec
    delay_samples = int(round(delay_sec * sr))
    return delay_samples

def mean_power_spectrum(path: str, start: int, length: int, sr: int,
                        n_fft: int, hop: int,
                        rms_gate_dbfs: float = -80.0):
    """
    计算一段音频的平均功率谱（rfft bins）
    - 先对齐后的重叠段里取 length 样点
    - 每帧：mono 能量平均 -> Hann -> rfft -> |X|^2
    - 对所有帧做平均
    - 可用 rms_gate_dbfs 丢掉极静音帧（避免把底噪/静音拉进统计）
    """
    x = read_segment(path, start, length, dtype="float32")
    assert x.shape[1] == 2, "只支持双声道"

    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (length - n_fft) // hop
    acc = None
    used = 0

    for i in range(n_frames):
        st = i * hop
        fr_lr = x[st:st + n_fft, :]
        mono = power_mono_lr(fr_lr)

        lv = rms_dbfs_mono(mono)
        if lv < rms_gate_dbfs:
            continue

        X = np.fft.rfft(mono * win)
        P = (X.real * X.real + X.imag * X.imag).astype(np.float64)

        if acc is None:
            acc = np.zeros_like(P, dtype=np.float64)
        acc += P
        used += 1

    if used == 0:
        raise RuntimeError("没有任何帧通过 rms_gate_dbfs 门限；请调低 --rms_gate_dbfs")

    acc /= float(used)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs.astype(np.float64), acc.astype(np.float64), used

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = int(win)
    if win <= 1:
        return x.copy()
    k = np.ones(win, dtype=np.float64) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(xpad, k, mode="valid")
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline FLAC")
    ap.add_argument("--cand", required=True, help="candidate FLAC")
    ap.add_argument("--out_csv", default="diff_spectrum.csv")
    ap.add_argument("--out_png", default="diff_vs_baseline.png")

    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--n_fft", type=int, default=8192)
    ap.add_argument("--hop", type=int, default=4096)

    ap.add_argument("--ds_sr", type=int, default=2000, help="对齐用的下采样率")
    ap.add_argument("--chunk_sec", type=float, default=25.0, help="对齐用 base chunk 秒数")
    ap.add_argument("--base_chunk_pos", choices=["mid", "start", "end"], default="mid")

    ap.add_argument("--max_minutes", type=float, default=10.0, help="用于频谱统计的最长分钟数（对齐后从起点开始）")
    ap.add_argument("--rms_gate_dbfs", type=float, default=-80.0, help="丢掉低于此阈值的帧（避免静音/底噪主导）")

    ap.add_argument("--anchor_lo", type=float, default=300.0, help="锚定频段下限（Hz），用于消除整体增益差")
    ap.add_argument("--anchor_hi", type=float, default=3000.0, help="锚定频段上限（Hz）")

    ap.add_argument("--clamp_lo", type=float, default=-12.0, help="ΔdB 限幅下限")
    ap.add_argument("--clamp_hi", type=float, default=+12.0, help="ΔdB 限幅上限")

    ap.add_argument("--smooth_win", type=int, default=31, help="平滑窗口（在 log 频率网格上做移动平均）")
    ap.add_argument("--log_grid_n", type=int, default=512, help="log 频率网格点数（用于平滑更稳）")

    args = ap.parse_args()

    # --- 0) 读元信息 ---
    with sf.SoundFile(args.base) as fb, sf.SoundFile(args.cand) as fc:
        assert fb.samplerate == args.sr and fc.samplerate == args.sr, "采样率必须一致"
        assert fb.channels == 2 and fc.channels == 2, "只支持双声道"
        Nbase = fb.frames
        Ncand = fc.frames

    # --- 1) 估计时间对齐 ---
    delay = find_delay_by_corr(args.cand, args.base, sr=args.sr, ds_sr=args.ds_sr,
                              chunk_sec=args.chunk_sec, base_chunk_pos=args.base_chunk_pos)
    print(f"[ALIGN] delay (cand - base) = {delay} samples = {delay/args.sr*1000:.2f} ms")

    base_start = max(0, -delay)
    cand_start = max(0, delay)

    max_len = int(args.max_minutes * 60.0 * args.sr)
    avail = min(Nbase - base_start, Ncand - cand_start, max_len)
    if avail <= args.n_fft:
        raise RuntimeError("可用重叠段太短，无法计算频谱。")

    # --- 2) 计算平均功率谱 ---
    freqs, P_base, used_b = mean_power_spectrum(args.base, base_start, avail, args.sr,
                                               args.n_fft, args.hop, args.rms_gate_dbfs)
    _,     P_cand, used_c = mean_power_spectrum(args.cand, cand_start, avail, args.sr,
                                               args.n_fft, args.hop, args.rms_gate_dbfs)
    print(f"[SPECTRUM] used frames: base={used_b}, cand={used_c}, seconds={avail/args.sr:.2f}")

    # --- 3) ΔdB = base - cand ---
    S_base_db = 10.0 * np.log10(P_base + EPS)
    S_cand_db = 10.0 * np.log10(P_cand + EPS)
    delta_db = (S_base_db - S_cand_db).astype(np.float64)

    # --- 4) anchor：把 300-3000Hz 的均值归零（消除整体增益差）---
    a0, a1 = args.anchor_lo, args.anchor_hi
    am = (freqs >= a0) & (freqs <= a1)
    if np.any(am):
        anchor = float(np.mean(delta_db[am]))
        delta_db = delta_db - anchor
        print(f"[ANCHOR] mean delta in {a0:.0f}-{a1:.0f}Hz = {anchor:.2f} dB (subtracted)")
    else:
        print("[ANCHOR] anchor band has no bins; skipped")

    # --- 5) clamp ---
    delta_db = np.clip(delta_db, args.clamp_lo, args.clamp_hi)

    # --- 6) 在 log 频率网格上做平滑（比直接在 rfft bins 上更符合听感尺度）---
    fmin = max(20.0, float(freqs[1]))
    fmax = float(freqs[-1])
    flog = np.logspace(np.log10(fmin), np.log10(fmax), args.log_grid_n).astype(np.float64)
    delta_log = np.interp(flog, freqs, delta_db).astype(np.float64)
    delta_smooth = moving_average(delta_log, args.smooth_win)

    # --- 7) 输出 CSV ---
    out = np.stack([flog, delta_log, delta_smooth], axis=1)
    np.savetxt(args.out_csv, out, delimiter=",",
               header="freq_hz,delta_db_raw,delta_db_smooth", comments="")
    print(f"[OUT] csv -> {args.out_csv}")

    # --- 8) 画图 ---
    plt.figure(figsize=(16, 7))
    plt.title("Candidate vs Baseline (Delta = base - cand)")
    plt.semilogx(flog, delta_log, label="raw")
    plt.semilogx(flog, delta_smooth, label=f"smooth (win={args.smooth_win})")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delta dB (base - candidate)")
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[OUT] png -> {args.out_png}")

if __name__ == "__main__":
    main()
