import argparse, json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve, medfilt

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

def kmeans2_1d(x: np.ndarray, iters=25):
    m1, m2 = np.percentile(x, [30, 70]).astype(float)
    for _ in range(iters):
        d1 = np.abs(x - m1)
        d2 = np.abs(x - m2)
        c1 = x[d1 <= d2]
        c2 = x[d1 >  d2]
        if len(c1) > 0: m1 = float(np.mean(c1))
        if len(c2) > 0: m2 = float(np.mean(c2))
    lab = (np.abs(x - m2) < np.abs(x - m1)).astype(np.int32)
    return lab, m1, m2

def find_delay_by_corr(orig_path, base_path, sr=48000, ds_sr=2000, chunk_sec=25):
    with sf.SoundFile(orig_path) as fo, sf.SoundFile(base_path) as fb:
        assert fo.samplerate == sr and fb.samplerate == sr
        assert fo.channels == 2 and fb.channels == 2
        Norig = fo.frames
        Nbase = fb.frames

        mid = int(0.5 * Nbase)
        half = int(0.5 * chunk_sec * sr)
        s = max(0, mid - half)
        e = min(Nbase, mid + half)

        fb.seek(s)
        xb = fb.read(e - s, dtype="float32", always_2d=True)
        mb = power_mono(xb)

        mb_ds = resample_poly(mb, ds_sr, sr).astype(np.float32)
        mb_ds = mb_ds - np.mean(mb_ds)

        fo.seek(0)
        mo_chunks = []
        block = sr * 30
        while True:
            x = fo.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            mo_chunks.append(power_mono(x))
        mo = np.concatenate(mo_chunks).astype(np.float32)
        mo_ds = resample_poly(mo, ds_sr, sr).astype(np.float32)
        mo_ds = mo_ds - np.mean(mo_ds)

    rb = mb_ds[::-1]
    corr = fftconvolve(mo_ds, rb, mode="valid")
    k = int(np.argmax(corr))
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

def debounce_state(state: np.ndarray, min_run: int = 3) -> np.ndarray:
    """
    最短持续帧约束：避免基准状态因为残余小爆点出现 1-2 帧抖动
    min_run=3 表示切换后至少持续 3 帧才承认
    """
    s = state.copy()
    n = len(s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        run = j - i
        if run < min_run:
            left = s[i-1] if i > 0 else s[j] if j < n else s[i]
            s[i:j] = left
        i = j
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--gate_ui", type=float, default=50.0)
    ap.add_argument("--gate_scale", type=float, default=1.0)

    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=2048)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--max_minutes", type=float, default=6.0)

    ap.add_argument("--hyst_list", type=float, nargs="+", default=[0,1,2,3,4,6])
    ap.add_argument("--delay_list_ms", type=float, nargs="+", default=[0,50,100,150,200,250])

    ap.add_argument("--tilt_lo", type=int, nargs=2, default=[200,1000])
    ap.add_argument("--tilt_hi", type=int, nargs=2, default=[2000,8000])
    ap.add_argument("--tilt_medfilt", type=int, default=5, help="tilt 中值滤波核大小(奇数), 3/5/7")

    ap.add_argument("--music_dbfs", type=float, default=-65.0, help="只在基准音量高于此阈值的帧上拟合")
    ap.add_argument("--gain_search_pm_db", type=float, default=3.0, help="围绕初始 gain_db0 的搜索范围 ±dB")
    ap.add_argument("--gain_step_db", type=float, default=0.5)

    ap.add_argument("--T_pm_db", type=float, default=10.0, help="围绕 T0 搜索范围 ±dB")
    ap.add_argument("--T_step_db", type=float, default=0.25)

    ap.add_argument("--out_json", default="calibration_v2.json")
    args = ap.parse_args()

    sr = args.sr
    delay = find_delay_by_corr(args.orig, args.base, sr=sr)
    print(f"[ALIGN] estimated delay (orig - base): {delay} samples ({delay/sr*1000:.2f} ms)")

    with sf.SoundFile(args.orig) as fo, sf.SoundFile(args.base) as fb:
        assert fo.samplerate == sr and fb.samplerate == sr
        assert fo.channels == 2 and fb.channels == 2

        base_start = max(0, -delay)
        orig_start = max(0,  delay)

        max_len = int(args.max_minutes * 60 * sr)
        avail = min(fb.frames - base_start, fo.frames - orig_start, max_len)
        if avail <= args.n_fft:
            raise ValueError("重叠可用长度太短，无法校准。")

        fb.seek(base_start)
        fo.seek(orig_start)
        xb = fb.read(avail, dtype="float32", always_2d=True)
        xo = fo.read(avail, dtype="float32", always_2d=True)

    n_fft = args.n_fft
    hop = args.hop
    n_frames = 1 + (avail - n_fft) // hop
    frame_starts = (np.arange(n_frames) * hop).astype(np.int64)

    orig_level = np.zeros(n_frames, np.float32)
    base_level = np.zeros(n_frames, np.float32)
    tilts = np.zeros(n_frames, np.float32)

    lo = tuple(args.tilt_lo)
    hi = tuple(args.tilt_hi)

    for i, st in enumerate(frame_starts):
        fo_fr = xo[st:st+n_fft, :]
        fb_fr = xb[st:st+n_fft, :]

        orig_level[i] = rms_dbfs_from_mono(power_mono(fo_fr))
        base_level[i] = rms_dbfs_from_mono(power_mono(fb_fr))
        tilts[i] = stft_band_tilt(fb_fr, sr, n_fft, lo=lo, hi=hi)

    # 仅用“有音乐的帧”做拟合（避免静音/极低电平放大噪声影响）
    music_mask = base_level > args.music_dbfs
    music_ratio = float(np.mean(music_mask))
    print(f"[MASK] music frames ratio: {music_ratio*100:.1f}% (threshold {args.music_dbfs} dBFS)")
    if music_ratio < 0.2:
        print("[WARN] 可用音乐帧太少，建议把 --music_dbfs 调低一点（例如 -70）")

    # tilt 平滑 + 聚类得到基准状态
    k = int(args.tilt_medfilt)
    if k % 2 == 0: k += 1
    if k < 3: k = 3
    tilts_s = medfilt(tilts, kernel_size=k).astype(np.float32)

    lab, _, _ = kmeans2_1d(tilts_s[music_mask])
    # 把 lab 映射回全长数组
    base_state = np.ones(n_frames, np.int32)
    base_state[music_mask] = np.where(lab==1, 2, 1).astype(np.int32)

    # 使“tilt 更高 => C2”
    mean1 = float(np.mean(tilts_s[music_mask][lab==1])) if np.any(lab==1) else -1e9
    mean0 = float(np.mean(tilts_s[music_mask][lab==0])) if np.any(lab==0) else -1e9
    if mean0 > mean1:
        base_state[music_mask] = np.where(lab==0, 2, 1).astype(np.int32)

    # 去抖（避免残余小爆点导致 1-2 帧切换）
    base_state = debounce_state(base_state, min_run=3)

    # 先估计“整体增益差”：base_level ≈ orig_level + gain_db
    gain_db0 = float(np.median((base_level - orig_level)[music_mask]))
    print(f"[GAIN] initial gain_db0 (base - orig): {gain_db0:.2f} dB")

    gains = np.arange(gain_db0 - args.gain_search_pm_db,
                      gain_db0 + args.gain_search_pm_db + 1e-9,
                      args.gain_step_db).astype(np.float32)

    best = None

    # 预先取出用于拟合的索引，减少无用帧
    idx = np.flatnonzero(music_mask)
    fs_fit = frame_starts[idx]

    for gain_db in gains:
        levels_adj = (orig_level + gain_db)[idx]  # 对应“把 orig 乘以 10^(gain_db/20)”后的 dBFS

        # 基于 base_state 的两类电平分布，估一个 T0 并缩小搜索范围
        s_fit = base_state[idx]
        c1 = levels_adj[s_fit == 1]
        c2 = levels_adj[s_fit == 2]
        if len(c1) < 10 or len(c2) < 10:
            continue
        T0 = 0.5 * (float(np.median(c1)) + float(np.median(c2)))

        Ts = np.arange(T0 - args.T_pm_db, T0 + args.T_pm_db + 1e-9, args.T_step_db).astype(np.float32)

        for up_ms in args.delay_list_ms:
            for hyst in args.hyst_list:
                for T in Ts:
                    pred = simulate_state(levels_adj, fs_fit, sr, float(T), float(hyst), float(up_ms))
                    mismatch = float(np.mean(pred != s_fit))
                    switches = int(np.sum(pred[1:] != pred[:-1]))
                    score = mismatch + 1e-5 * switches  # 轻微惩罚过度抖动

                    if best is None or score < best["score"]:
                        best = dict(score=score, mismatch=mismatch, switches=switches,
                                    T=float(T), hyst=float(hyst), up_ms=float(up_ms),
                                    gain_db=float(gain_db), T0=float(T0))

    if best is None:
        raise RuntimeError("未找到可用最优解：请放宽 --music_dbfs 或增大 --max_minutes。")

    # 重要：T 是在“调整后 levels_adj”上的阈值。
    # process_tomatis 用的是原始音频（未加 gain），所以阈值要还原：
    # levels_adj = levels_raw + gain_db  =>  levels_raw 与阈值关系等价于：T_raw = T - gain_db
    T_adj = best["T"]
    gain_db = best["gain_db"]
    T_raw = T_adj - gain_db

    gate_offset = T_raw - args.gate_scale * args.gate_ui

    print("\n[BEST]")
    print(best)
    print(f"\n[RECOMMEND] gain_db (diagnostic only): {gain_db:+.2f} dB (base - orig)")
    print(f"[RECOMMEND] T_adj (on leveled orig): {T_adj:.2f} dBFS")
    print(f"[RECOMMEND] T_raw (for process_tomatis): {T_raw:.2f} dBFS")
    print(f"[RECOMMEND] gate_ui={args.gate_ui:.1f}, gate_scale={args.gate_scale:.2f}, gate_offset={gate_offset:.2f}")
    print(f"[RECOMMEND] hyst_db={best['hyst']:.1f}, up_delay_ms={best['up_ms']:.0f}")
    print(f"[RECOMMEND] mismatch={best['mismatch']*100:.2f}%, switches={best['switches']} (on music frames)")

    out = {
        "orig": args.orig,
        "base": args.base,
        "delay_samples_orig_minus_base": int(delay),
        "music_dbfs": float(args.music_dbfs),

        "gain_db_base_minus_orig": float(gain_db),  # 诊断用
        "T_adj_dbfs": float(T_adj),                 # 诊断用
        "T_raw_dbfs": float(T_raw),                 # 用于实际处理

        "gate_ui": float(args.gate_ui),
        "gate_scale": float(args.gate_scale),
        "gate_offset": float(gate_offset),

        "hyst_db": float(best["hyst"]),
        "up_delay_ms": float(best["up_ms"]),

        "mismatch": float(best["mismatch"]),
        "switches": int(best["switches"])
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {args.out_json}")

if __name__ == "__main__":
    main()
