import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import convolve

EPS = 1e-12

def db_to_lin(db):
    return 10.0 ** (db / 20.0)

def smooth_on_logfreq(freq, db, win=21):
    # 在 log-frequency 轴做简单移动平均
    lf = np.log10(np.maximum(freq, 1.0))
    order = np.argsort(lf)
    lf2 = lf[order]
    db2 = db[order]

    # 重新采样到等间隔 log 轴
    n = len(db2)
    lf_grid = np.linspace(lf2.min(), lf2.max(), n)
    db_grid = np.interp(lf_grid, lf2, db2)

    # 移动平均
    win = max(3, win | 1)  # 强制奇数
    pad = win // 2
    x = np.pad(db_grid, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    y = np.convolve(x, kernel, mode="valid")

    # 映射回原频率点
    db_smooth = np.interp(lf2, lf_grid, y)
    out = np.empty_like(db_smooth)
    out[order] = db_smooth
    return out

def build_eq_from_residual(freqs_rfft, res_freq, res_db,
                           clamp_lo=-6.0, clamp_hi=6.0,
                           mid_start=3000.0, mid_clamp_hi=2.0,
                           hf_start=8000.0, hf_clamp_hi=0.0):
    # 插值到 FFT bins
    db = np.interp(freqs_rfft, res_freq, res_db, left=res_db[0], right=res_db[-1])

    # 全局限幅
    db = np.clip(db, clamp_lo, clamp_hi)

    # 中高频限幅 (3k-8k)
    mid_mask = (freqs_rfft >= mid_start) & (freqs_rfft < hf_start)
    db[mid_mask] = np.clip(db[mid_mask], clamp_lo, mid_clamp_hi)

    # 高频限幅 (8k+)
    hf_mask = freqs_rfft >= hf_start
    db[hf_mask] = np.clip(db[hf_mask], clamp_lo, hf_clamp_hi)

    return db_to_lin(db).astype(np.float32), db.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_audio", required=True, help="候选音频（例如 D_MNF_matched_v2_eq_gp.flac）")
    ap.add_argument("--out_audio", required=True, help="输出音频")
    ap.add_argument("--diff_csv", default="diff_spectrum.csv", help="对比生成的 diff_spectrum.csv")
    ap.add_argument("--n_fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=2048)
    ap.add_argument("--smooth_win", type=int, default=41)
    ap.add_argument("--clamp_hi", type=float, default=6.0)
    ap.add_argument("--mid_start", type=float, default=3000.0)
    ap.add_argument("--mid_clamp_hi", type=float, default=2.0)
    ap.add_argument("--hf_start", type=float, default=8000.0)
    ap.add_argument("--hf_clamp_hi", type=float, default=0.0)
    args = ap.parse_args()

    diff = pd.read_csv(args.diff_csv)
    res_freq = diff["freq_hz"].to_numpy(np.float32)
    # 兼容旧版 CSV 列名
    col = "delta_db_base_minus_cand" if "delta_db_base_minus_cand" in diff.columns else "delta_db"
    res_db = diff[col].to_numpy(np.float32)

    # 平滑残差曲线
    res_db_s = smooth_on_logfreq(res_freq, res_db, win=args.smooth_win)

    with sf.SoundFile(args.in_audio) as f:
        sr = f.samplerate
        ch = f.channels
        assert ch == 2, "只支持双声道"

    n_fft = args.n_fft
    hop = args.hop
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    eq_lin, eq_db = build_eq_from_residual(
        freqs, res_freq, res_db_s,
        clamp_lo=-6.0, clamp_hi=args.clamp_hi,
        mid_start=args.mid_start, mid_clamp_hi=args.mid_clamp_hi,
        hf_start=args.hf_start, hf_clamp_hi=args.hf_clamp_hi
    )

    win = np.hanning(n_fft).astype(np.float32)
    win2 = (win * win).astype(np.float32)

    with sf.SoundFile(args.in_audio) as fin, sf.SoundFile(args.out_audio, "w",
                                                         samplerate=sr, channels=2,
                                                         format="FLAC", subtype="PCM_24") as fout:
        in_buf = np.zeros((0, 2), np.float32)
        in_base = 0
        next_start = 0

        out_buf = np.zeros((0, 2), np.float32)
        w_buf = np.zeros((0,), np.float32)
        out_base = 0

        def ensure_out(end_pos):
            nonlocal out_buf, w_buf, out_base
            need = end_pos - out_base
            if need <= len(w_buf): return
            grow = need - len(w_buf)
            out_buf = np.vstack([out_buf, np.zeros((grow, 2), np.float32)])
            w_buf = np.concatenate([w_buf, np.zeros((grow,), np.float32)])

        block = sr * 10
        processed_frames = 0
        while True:
            x = fin.read(block, dtype="float32", always_2d=True)
            if len(x) == 0: break
            in_buf = np.vstack([in_buf, x])

            while True:
                rel = next_start - in_base
                if rel + n_fft > len(in_buf):
                    break
                frame = in_buf[rel:rel+n_fft, :]

                y = np.zeros_like(frame, np.float32)
                for c in range(2):
                    X = np.fft.rfft(frame[:, c] * win)
                    X *= eq_lin
                    y[:, c] = np.fft.irfft(X, n=n_fft).astype(np.float32) * win

                start = next_start
                end = start + n_fft
                ensure_out(end)
                orel = start - out_base
                out_buf[orel:orel+n_fft, :] += y
                w_buf[orel:orel+n_fft] += win2

                next_start += hop

                safe = next_start - out_base
                if safe >= sr * 5:
                    n = safe
                    fout.write(out_buf[:n, :] / (w_buf[:n, None] + EPS))
                    out_base += n
                    out_buf = out_buf[n:, :]
                    w_buf = w_buf[n:]

            keep = max(0, len(in_buf) - n_fft)
            if keep > 0:
                in_buf = in_buf[keep:, :]
                in_base += keep

        if len(w_buf) > 0:
            fout.write(out_buf / (w_buf[:, None] + EPS))
    
    print(f"[DONE] Applied residual EQ to {args.out_audio}")

if __name__ == "__main__":
    main()
