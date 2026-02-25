import argparse
import numpy as np
import soundfile as sf
import csv

EPS = 1e-12

def db_to_lin(db):
    return 10.0 ** (db / 20.0)

def load_eq_csv(eq_csv_path):
    """
    期望 CSV 至少两列：freq_hz, delta_db
    也兼容列名为 freq, hz, f / delta, db, gain_db, delta_db_smooth 等
    """
    freqs = []
    dbs = []
    with open(eq_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.lower().strip() for c in reader.fieldnames]

        def pick(name_candidates):
            for cand in name_candidates:
                if cand in cols:
                    return cand
            return None

        f_col = pick(["freq_hz", "freq", "hz", "f"])
        # 优先找 smooth，找不到找 raw/delta_db
        d_col = pick(["delta_db_smooth", "delta_db", "db", "gain_db", "delta", "gain"])

        if f_col is None or d_col is None:
            raise ValueError(f"EQ CSV 列名不符合预期。发现列: {reader.fieldnames}")

        print(f"[EQ_LOAD] Using columns: freq='{f_col}', gain='{d_col}'")

        for row in reader:
            freqs.append(float(row[f_col]))
            dbs.append(float(row[d_col]))

    freqs = np.array(freqs, np.float32)
    dbs = np.array(dbs, np.float32)

    # 排序，防止插值异常
    idx = np.argsort(freqs)
    return freqs[idx], dbs[idx]

def build_gain_per_bin(sr, n_fft, eq_freqs, eq_db):
    """
    把 (freq->dB) 插值到 rfft bins。
    插值在 log-f 轴上更稳定（接近听感）。
    """
    f_bins = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)
    f_safe = np.maximum(f_bins, 1.0)

    # log-f 插值
    x = np.log10(np.maximum(eq_freqs, 1.0))
    y = eq_db
    xb = np.log10(f_safe)

    # 超出范围：夹到边界值（避免极端）
    yb = np.interp(xb, x, y, left=y[0], right=y[-1]).astype(np.float32)
    gain = db_to_lin(yb).astype(np.float32)
    return gain

def apply_eq_stft(
    in_path,
    out_path,
    eq_csv,
    n_fft=4096,
    hop=2048,
    pad=True,
    global_gain_db=0.0,
    auto_gain_protect=True,
    peak_target=0.99,
):
    # --- 读输入 ---
    with sf.SoundFile(in_path, "r") as fin:
        sr = fin.samplerate
        ch = fin.channels
        if sr != 48000:
            raise ValueError(f"期望 48kHz，实际 {sr}")
        if ch != 2:
            raise ValueError(f"期望双声道，实际 {ch}")

        N = fin.frames

        # --- 输出 ---
        try:
            fout = sf.SoundFile(out_path, "w", samplerate=sr, channels=ch, format="FLAC", subtype="PCM_24")
            out_is_flac = True
            wav_fallback = None
        except Exception as e:
            wav_fallback = out_path.replace(".flac", ".wav")
            fout = sf.SoundFile(wav_fallback, "w", samplerate=sr, channels=ch, format="WAV", subtype="PCM_24")
            out_is_flac = False
            print(f"[WARN] FLAC 写入失败，先写 WAV: {e}")

        # --- EQ 曲线 -> 每 bin 增益 ---
        eq_freqs, eq_db = load_eq_csv(eq_csv)
        gain_bins = build_gain_per_bin(sr, n_fft, eq_freqs, eq_db)

        # --- 窗与 OLA ---
        win = np.hanning(n_fft).astype(np.float32)
        win2 = (win * win).astype(np.float32)

        pad_len = n_fft // 2 if pad else 0
        total_len = N + 2 * pad_len

        # 先整体增益（可选）
        g_global = db_to_lin(global_gain_db)

        # 流式处理：块读
        block = sr * 10  # 10s
        in_buf = np.zeros((0, ch), np.float32)
        in_base = 0
        next_start = 0

        out_buf = np.zeros((0, ch), np.float32)
        w_buf = np.zeros((0,), np.float32)
        out_base = 0

        # 预先插入前 padding（零）
        if pad_len > 0:
            in_buf = np.vstack([in_buf, np.zeros((pad_len, ch), np.float32)])
            in_base = 0
            next_start = 0

        def ensure_out(end_pos):
            nonlocal out_buf, w_buf, out_base
            need = end_pos - out_base
            if need <= len(w_buf):
                return
            grow = need - len(w_buf)
            out_buf = np.vstack([out_buf, np.zeros((grow, ch), np.float32)])
            w_buf = np.concatenate([w_buf, np.zeros((grow,), np.float32)])

        # 峰值保护：分段估计 peak
        peak_seen = 0.0

        # 处理循环
        frames_done = 0
        while True:
            x = fin.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            x = (x * g_global).astype(np.float32)
            in_buf = np.vstack([in_buf, x])

            while True:
                rel = next_start - in_base
                if rel + n_fft > len(in_buf):
                    break
                frame = in_buf[rel:rel+n_fft, :]

                # STFT EQ（左右声道同一条增益）
                y = np.zeros_like(frame, np.float32)
                for c in range(ch):
                    X = np.fft.rfft(frame[:, c] * win)
                    X *= gain_bins
                    y[:, c] = np.fft.irfft(X, n=n_fft).astype(np.float32) * win

                start = next_start
                end = start + n_fft
                ensure_out(end)
                orel = start - out_base
                out_buf[orel:orel+n_fft, :] += y
                w_buf[orel:orel+n_fft] += win2

                next_start += hop
                frames_done += 1

                # 每写出一段，更新 peak
                safe = next_start - out_base
                if safe >= sr * 5:
                    n = safe
                    seg = out_buf[:n, :] / (w_buf[:n, None] + EPS)
                    peak_seen = max(peak_seen, float(np.max(np.abs(seg))))
                    fout.write(seg)
                    out_base += n
                    out_buf = out_buf[n:, :]
                    w_buf = w_buf[n:]

        # 追加尾 padding（零），保证边界一致
        if pad_len > 0:
            in_buf = np.vstack([in_buf, np.zeros((pad_len, ch), np.float32)])
            # 把剩余帧刷完
            while True:
                rel = next_start - in_base
                if rel + n_fft > len(in_buf):
                    break
                frame = in_buf[rel:rel+n_fft, :]

                y = np.zeros_like(frame, np.float32)
                for c in range(ch):
                    X = np.fft.rfft(frame[:, c] * win)
                    X *= gain_bins
                    y[:, c] = np.fft.irfft(X, n=n_fft).astype(np.float32) * win

                start = next_start
                end = start + n_fft
                ensure_out(end)
                orel = start - out_base
                out_buf[orel:orel+n_fft, :] += y
                w_buf[orel:orel+n_fft] += win2

                next_start += hop
                frames_done += 1

        # 收尾写出
        if len(w_buf) > 0:
            seg = out_buf / (w_buf[:, None] + EPS)
            peak_seen = max(peak_seen, float(np.max(np.abs(seg))))
            fout.write(seg)

        fout.close()

    # 如果需要自动峰值保护：这里给“二遍整体缩放”的方案（最稳但要重写文件）
    # 你若不想二遍，就把 auto_gain_protect=False。
    if auto_gain_protect and peak_seen > peak_target:
        scale = peak_target / max(peak_seen, EPS)
        print(f"[GAIN_PROTECT] peak={peak_seen:.4f} > {peak_target}, apply scale={scale:.4f}")
        # 直接用 ffmpeg/sox 也行；这里给纯 python 再写一遍
        tmp_in = out_path if out_is_flac else wav_fallback
        tmp_out = out_path.replace(".flac", "_gp.flac")
        # 优化：流式读写
        with sf.SoundFile(tmp_in, "r") as fin, sf.SoundFile(tmp_out, "w", samplerate=fin.samplerate, channels=fin.channels, format="FLAC", subtype="PCM_24") as fout:
            while True:
                x = fin.read(48000*30, dtype="float32", always_2d=True)
                if len(x) == 0:
                    break
                fout.write((x * scale).astype(np.float32))
        print(f"[DONE] gain-protected file: {tmp_out}")

    print("[DONE] EQ applied.")
    if not out_is_flac and wav_fallback is not None:
        print(f'[NOTE] 输出为 WAV: {wav_fallback}，可用 ffmpeg 转 FLAC。')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--eq_csv", required=True)
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=2048)
    ap.add_argument("--no_pad", action="store_true")
    ap.add_argument("--gain_db", type=float, default=0.0, help="额外整体增益（dB），想完全匹配录音电平可用 -17.77")
    ap.add_argument("--no_gain_protect", action="store_true")
    args = ap.parse_args()

    apply_eq_stft(
        args.input,
        args.output,
        args.eq_csv,
        n_fft=args.n_fft,
        hop=args.hop,
        pad=(not args.no_pad),
        global_gain_db=args.gain_db,
        auto_gain_protect=(not args.no_gain_protect),
    )

if __name__ == "__main__":
    main()
