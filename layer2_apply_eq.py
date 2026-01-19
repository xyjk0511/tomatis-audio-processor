import argparse
import numpy as np
import soundfile as sf
from scipy.signal import firwin2

EPS = 1e-12

def load_curve_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    freqs = data[:, 0].astype(np.float64)
    delta_db = data[:, 2].astype(np.float64)  # 使用 smooth 曲线
    return freqs, delta_db

def design_fir_eq(freqs_hz, delta_db, sr, numtaps=8193, fmin=20.0, fmax=20000.0, headroom_db=1.0):
    """
    用 firwin2 设计线性相位 FIR：
    gain(f) = 10^(delta_db/20)，并加入少量 headroom 避免削波风险。
    """
    nyq = sr / 2.0

    # 只取指定频段，其他频段设为 0dB（增益=1）
    mask = (freqs_hz >= fmin) & (freqs_hz <= min(fmax, nyq))
    f = freqs_hz[mask]
    d = delta_db[mask]

    # 频率点必须从 0 到 nyq 单调递增
    # 端点：0Hz 用第一个点的增益；Nyquist 用最后一个点的增益（或 1.0）
    g = 10.0 ** (d / 20.0)

    # headroom：整体缩小一点，避免 EQ 后峰值顶爆
    g *= 10.0 ** (-headroom_db / 20.0)

    # 构造 firwin2 输入
    freq_pts = np.concatenate(([0.0], f, [nyq]))
    gain_pts = np.concatenate(([g[0]], g, [g[-1]]))

    # 归一化到 [0,1]
    freq_norm = freq_pts / nyq

    # 设计 FIR（线性相位、稳定）
    h = firwin2(numtaps=numtaps, freq=freq_norm, gain=gain_pts, window="hann")
    return h.astype(np.float32)

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def apply_fir_stream(in_path, out_path, h, block_samples=48000*10):
    # 打开输入
    fin = sf.SoundFile(in_path, "r")
    sr = fin.samplerate
    ch = fin.channels
    assert ch == 2, "只实现了双声道"
    assert sr == 48000, "期望 48kHz"

    # 尝试写 FLAC，失败则写 WAV
    try:
        fout = sf.SoundFile(out_path, "w", samplerate=sr, channels=ch, format="FLAC", subtype="PCM_24")
        out_is_flac = True
        wav_fallback = None
    except Exception as e:
        wav_fallback = out_path.replace(".flac", ".wav")
        fout = sf.SoundFile(wav_fallback, "w", samplerate=sr, channels=ch, format="WAV", subtype="PCM_24")
        out_is_flac = False
        print(f"[WARN] FLAC write failed: {e}")
        print(f"[WARN] Writing WAV instead: {wav_fallback}")

    M = len(h)
    overlap = M - 1

    # overlap-save FFT 设置
    L = block_samples
    Nfft = next_pow2(L + overlap)
    H = np.fft.rfft(h, n=Nfft).astype(np.complex64)

    # 每通道各自 overlap 缓冲
    prev = np.zeros((overlap, ch), dtype=np.float32)

    total = fin.frames
    processed = 0

    while True:
        x = fin.read(L, dtype="float32", always_2d=True)
        if len(x) == 0:
            break

        # 拼接 overlap
        x_in = np.vstack([prev, x])  # [overlap+L, ch]
        # padding 到 Nfft
        if len(x_in) < Nfft:
            x_pad = np.vstack([x_in, np.zeros((Nfft - len(x_in), ch), np.float32)])
        else:
            x_pad = x_in

        y_out = np.zeros((L, ch), dtype=np.float32)

        for c in range(ch):
            X = np.fft.rfft(x_pad[:, c], n=Nfft)
            Y = X * H
            y = np.fft.irfft(Y, n=Nfft).astype(np.float32)

            # overlap-save：丢掉前 overlap 样点，取后 L 样点
            y_out[:, c] = y[overlap:overlap+L]

        fout.write(y_out)

        # 更新 overlap
        if len(x_in) >= overlap:
            prev = x_in[-overlap:, :]
        else:
            prev = x_in

        processed += len(x)
        if processed % (48000 * 60) < L:  # 每分钟打一次
            print(f"[PROGRESS] {processed/sr:.1f}s / {total/sr:.1f}s ({processed/total*100:.1f}%)")

    fin.close()
    fout.close()

    if not out_is_flac and wav_fallback:
        print("\n[NOTE] 已输出 WAV（Windows 上 libsndfile 写 FLAC 可能失败）")
        print("用 ffmpeg 转 FLAC：")
        print(f'ffmpeg -y -i "{wav_fallback}" -c:a flac -compression_level 8 "{out_path}"')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_flac", required=True)
    ap.add_argument("--out_flac", required=True)
    ap.add_argument("--curve_csv", required=True)
    ap.add_argument("--numtaps", type=int, default=8193)
    ap.add_argument("--headroom_db", type=float, default=1.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=20000.0)
    ap.add_argument("--block_sec", type=float, default=10.0)
    args = ap.parse_args()

    freqs, delta_db = load_curve_csv(args.curve_csv)
    h = design_fir_eq(freqs, delta_db, sr=48000, numtaps=args.numtaps,
                      fmin=args.fmin, fmax=args.fmax, headroom_db=args.headroom_db)

    print(f"[FIR] numtaps={len(h)}, headroom_db={args.headroom_db}")
    apply_fir_stream(args.in_flac, args.out_flac, h, block_samples=int(48000 * args.block_sec))
    print(f"[DONE] {args.out_flac}")

if __name__ == "__main__":
    main()
