import argparse, numpy as np, soundfile as sf
from scipy.signal import resample_poly, fftconvolve

EPS = 1e-12

def power_mono(x_lr: np.ndarray) -> np.ndarray:
    # x_lr: [N,2] -> mono amplitude via power average
    p = 0.5 * (x_lr[:,0]*x_lr[:,0] + x_lr[:,1]*x_lr[:,1])
    return np.sqrt(p + EPS)

def rms_dbfs_from_mono(mono: np.ndarray) -> float:
    r = np.sqrt(np.mean(mono*mono) + EPS)
    return float(20*np.log10(r + EPS))

def stft_band_tilt(frame_lr: np.ndarray, sr: int, n_fft: int,
                   lo=(200,1000), hi=(2000,8000)) -> float:
    # 用“高频能量/低频能量”的 log 比值当作 C1/C2 判别特征
    win = np.hanning(n_fft).astype(np.float32)
    mono = power_mono(frame_lr) * win
    X = np.fft.rfft(mono)
    P = (X.real*X.real + X.imag*X.imag).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    lo_mask = (freqs>=lo[0]) & (freqs<lo[1])
    hi_mask = (freqs>=hi[0]) & (freqs<hi[1])

    Elo = float(np.sum(P[lo_mask]) + EPS)
    Ehi = float(np.sum(P[hi_mask]) + EPS)
    return float(10*np.log10(Ehi/Elo + EPS))

def kmeans2_1d(x: np.ndarray, iters=20):
    # 简易 1D kmeans (k=2)，不依赖 sklearn
    m1, m2 = np.percentile(x, [30, 70]).astype(float)
    for _ in range(iters):
        d1 = np.abs(x - m1)
        d2 = np.abs(x - m2)
        c1 = x[d1 <= d2]
        c2 = x[d1 > d2]
        if len(c1) > 0: m1 = float(np.mean(c1))
        if len(c2) > 0: m2 = float(np.mean(c2))
    # 返回每点属于哪个簇（0/1），以及两个簇均值
    lab = (np.abs(x - m2) < np.abs(x - m1)).astype(np.int32)
    return lab, m1, m2

def find_delay_by_corr(orig_path, base_path, sr=48000, ds_sr=2000, chunk_sec=25):
    """
    用“基准中间一段 chunk”与“原始全长”做 FFT 互相关，估计全局延迟（样点）。
    """
    with sf.SoundFile(orig_path) as fo, sf.SoundFile(base_path) as fb:
        assert fo.samplerate == sr and fb.samplerate == sr
        assert fo.channels == 2 and fb.channels == 2
        Norig = fo.frames
        Nbase = fb.frames

        # 取基准中间一段有音乐的 chunk（默认 25s）
        mid = int(0.5 * Nbase)
        half = int(0.5 * chunk_sec * sr)
        s = max(0, mid - half)
        e = min(Nbase, mid + half)
        fb.seek(s)
        xb = fb.read(e - s, dtype="float32", always_2d=True)
        mb = power_mono(xb)

        # 下采样到 ds_sr（比如 2kHz）降低相关计算量
        up = ds_sr
        down = sr
        mb_ds = resample_poly(mb, up, down).astype(np.float32)
        mb_ds = mb_ds - np.mean(mb_ds)

        # 读原始全长（mono，下采样）
        fo.seek(0)
        mo_chunks = []
        block = sr * 30  # 30s
        while True:
            x = fo.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            mo_chunks.append(power_mono(x))
        mo = np.concatenate(mo_chunks).astype(np.float32)
        mo_ds = resample_poly(mo, up, down).astype(np.float32)
        mo_ds = mo_ds - np.mean(mo_ds)

    # 互相关：corr[k] = sum orig[t]*base[t-k]
    # 用 fftconvolve(orig, reverse(base))
    rb = mb_ds[::-1]
    corr = fftconvolve(mo_ds, rb, mode="valid")
    k = int(np.argmax(corr))  # base_chunk 对齐到 orig 的位置（下采样坐标）
    # 把“chunk中心点”映射到 orig
    base_center = (s + (e - s)//2) / sr
    orig_center = (k + len(mb_ds)//2) / ds_sr
    delay_sec = orig_center - base_center
    delay_samples = int(round(delay_sec * sr))
    return delay_samples

def simulate_state(level_dbfs: np.ndarray, frame_starts: np.ndarray, sr: int,
                   T: float, hyst: float, up_delay_ms: float):
    Ton = T + hyst/2
    Toff = T - hyst/2
    up_delay_samples = int(round(sr * up_delay_ms / 1000.0))

    state = 1
    pending = None
    out = np.zeros_like(level_dbfs, dtype=np.int32)

    for i, (lv, st) in enumerate(zip(level_dbfs, frame_starts)):
        if state == 1:
            if lv >= Ton:
                if pending is None:
                    pending = st + up_delay_samples
            else:
                pending = None
            if pending is not None and st >= pending:
                state = 2
                pending = None
        else:
            if lv <= Toff:
                state = 1
                pending = None
        out[i] = state
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="原始 D MNF FLAC")
    ap.add_argument("--base", required=True, help="基准 Tomatis_D_final FLAC")
    ap.add_argument("--gate_ui", type=float, default=50.0)
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=2048)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--search_T_min", type=float, default=-40.0)
    ap.add_argument("--search_T_max", type=float, default=-10.0)
    ap.add_argument("--search_T_step", type=float, default=0.5)
    ap.add_argument("--hyst_list", type=float, nargs="+", default=[0,1,2,3,4,6,8])
    ap.add_argument("--delay_list_ms", type=float, nargs="+", default=[0,50,100,150,200,250,300])
    ap.add_argument("--max_minutes", type=float, default=8.0, help="用于校准的最长分钟数（从对齐起点开始截取）")
    args = ap.parse_args()

    sr = args.sr
    # 1) 估计全局时间对齐 delay（样点，orig 相对 base）
    delay = find_delay_by_corr(args.orig, args.base, sr=sr)
    print(f"[ALIGN] estimated delay (orig - base): {delay} samples ({delay/sr*1000:.2f} ms)")

    # 2) 读对齐后的重叠片段（只取前 max_minutes 以加速）
    with sf.SoundFile(args.orig) as fo, sf.SoundFile(args.base) as fb:
        assert fo.samplerate == sr and fb.samplerate == sr
        assert fo.channels == 2 and fb.channels == 2

        Norig = fo.frames
        Nbase = fb.frames

        # base 从 0 开始，orig 从 delay 开始（若 delay<0，说明 base 比 orig 晚）
        base_start = max(0, -delay)
        orig_start = max(0, delay)

        max_len = int(args.max_minutes * 60 * sr)
        avail = min(Nbase - base_start, Norig - orig_start, max_len)
        if avail <= args.n_fft:
            raise ValueError("重叠可用长度太短，无法校准。请检查是否为同一首音频或基准裁剪过短。")

        fb.seek(base_start)
        fo.seek(orig_start)
        xb = fb.read(avail, dtype="float32", always_2d=True)
        xo = fo.read(avail, dtype="float32", always_2d=True)

    # 3) 逐帧计算：orig 的 level(dBFS)；base 的 tilt 特征并聚类得到“基准状态”
    n_fft = args.n_fft
    hop = args.hop
    n_frames = 1 + (avail - n_fft) // hop
    frame_starts = (np.arange(n_frames) * hop).astype(np.int64)

    levels = np.zeros(n_frames, np.float32)
    tilts = np.zeros(n_frames, np.float32)

    for i, st in enumerate(frame_starts):
        fo_fr = xo[st:st+n_fft, :]
        fb_fr = xb[st:st+n_fft, :]

        mono_o = power_mono(fo_fr)
        levels[i] = rms_dbfs_from_mono(mono_o)
        tilts[i] = stft_band_tilt(fb_fr, sr, n_fft)

    # 基准状态：tilt 两簇；tilt 更高 -> C2（高频更亮）
    lab, m1, m2 = kmeans2_1d(tilts)
    mean0 = float(np.mean(tilts[lab==0])) if np.any(lab==0) else -1e9
    mean1 = float(np.mean(tilts[lab==1])) if np.any(lab==1) else -1e9
    # 令 base_state: 1=C1, 2=C2
    if mean1 > mean0:
        base_state = np.where(lab==1, 2, 1).astype(np.int32)
    else:
        base_state = np.where(lab==0, 2, 1).astype(np.int32)

    print(f"[BASE] tilt cluster means: {mean0:.3f}, {mean1:.3f} (higher => C2)")

    # 4) 网格搜索 T/hyst/up_delay 使模拟状态最接近基准状态
    Ts = np.arange(args.search_T_min, args.search_T_max + 1e-9, args.search_T_step)
    best = None

    for up_ms in args.delay_list_ms:
        for hyst in args.hyst_list:
            for T in Ts:
                pred = simulate_state(levels, frame_starts, sr, T, hyst, up_ms)
                mismatch = float(np.mean(pred != base_state))
                # 轻微惩罚：避免过度抖动（切换太频繁）
                switches = int(np.sum(pred[1:] != pred[:-1]))
                score = mismatch + 1e-6 * switches
                if best is None or score < best["score"]:
                    best = dict(score=score, mismatch=mismatch, switches=switches, T=T, hyst=hyst, up_ms=up_ms)

    print("[BEST]")
    print(best)
    # 5) 输出建议参数（固定 gate_scale=1：offset = T - gate_ui）
    gate_offset = best["T"] - args.gate_ui
    print(f"[RECOMMEND] gate_ui={args.gate_ui:.1f}, gate_scale=1.0, gate_offset={gate_offset:.2f}")
    print(f"[RECOMMEND] hyst_db={best['hyst']:.1f}, up_delay_ms={best['up_ms']:.0f}")
    print(f"[RECOMMEND] (overlap segment) mismatch={best['mismatch']*100:.2f}%, switches={best['switches']}")

    import json
    res = {
        "gate_offset": gate_offset,
        "hyst_db": best['hyst'],
        "up_delay_ms": best['up_ms'],
        "gate_ui": args.gate_ui
    }
    with open("calibration.json", "w") as f:
        json.dump(res, f)

if __name__ == "__main__":
    main()
