"""
验证倾斜滤波器的实际增益幅度

理论值 (fc=1000Hz, slope=12dB/oct, +/-15dB):
- 250Hz: +/-15dB (距1000Hz 2个octave, 2*12=24 > 15, 所以平台)
- 1000Hz: 0dB (中心频率)
- 4000Hz: +/-15dB (距1000Hz 2个octave)

C1: 250Hz +15dB, 4000Hz -15dB -> 倾斜 = -30dB
C2: 250Hz -15dB, 4000Hz +15dB -> 倾斜 = +30dB
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve

EPS = 1e-12
SR = 48000
N_FFT = 4096
HOP = 2048

def power_mono(x_lr):
    p = 0.5 * (x_lr[:,0]**2 + x_lr[:,1]**2)
    return np.sqrt(p + EPS)

def find_delay_xcorr(base_mono, cand_mono, sr=SR, ds_sr=2000):
    up, down = ds_sr, sr
    b = resample_poly(base_mono - base_mono.mean(), up, down).astype(np.float32)
    c = resample_poly(cand_mono - cand_mono.mean(), up, down).astype(np.float32)
    corr = fftconvolve(c, b[::-1], mode="full")
    k = int(np.argmax(corr))
    shift_ds = k - (len(b) - 1)
    return int(round(shift_ds * (sr / ds_sr)))

def align_audio(base_lr, cand_lr, delay):
    if delay > 0:
        cand_lr = cand_lr[delay:]
    elif delay < 0:
        base_lr = base_lr[-delay:]
    n = min(len(base_lr), len(cand_lr))
    return base_lr[:n], cand_lr[:n]

def band_power_db(spec_db, freqs, f1, f2):
    """计算频段平均功率(dB)"""
    mask = (freqs >= f1) & (freqs < f2)
    return np.mean(spec_db[mask])

def analyze_tilt_detail(input_path, output_path):
    print("=" * 70)
    print("倾斜滤波器增益幅度验证")
    print("=" * 70)

    # 读取并对齐
    inp_lr, _ = sf.read(input_path, dtype="float32", always_2d=True)
    out_lr, _ = sf.read(output_path, dtype="float32", always_2d=True)

    inp_mono = power_mono(inp_lr)
    out_mono = power_mono(out_lr)

    delay = find_delay_xcorr(inp_mono, out_mono)
    print(f"Delay: {delay} samples")

    inp_aligned, out_aligned = align_audio(inp_lr, out_lr, delay)

    # STFT
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1/SR)
    n_frames = 1 + (len(inp_aligned) - N_FFT) // HOP

    # 收集C1和C2帧的频谱差
    c1_diffs = []  # 低电平帧 (< -45 dBFS)
    c2_diffs = []  # 高电平帧 (> -30 dBFS)

    for i in range(n_frames):
        st = i * HOP
        inp_frame = power_mono(inp_aligned[st:st+N_FFT])
        out_frame = power_mono(out_aligned[st:st+N_FFT])

        inp_level = 20 * np.log10(np.sqrt(np.mean(inp_frame**2) + EPS) + EPS)

        # 计算频谱
        inp_spec = np.abs(np.fft.rfft(inp_frame * win)) + EPS
        out_spec = np.abs(np.fft.rfft(out_frame * win)) + EPS

        inp_db = 20 * np.log10(inp_spec)
        out_db = 20 * np.log10(out_spec)
        diff_db = out_db - inp_db

        if inp_level < -45:
            c1_diffs.append(diff_db)
        elif inp_level > -30:
            c2_diffs.append(diff_db)

    print(f"\nC1 frames (level < -45 dBFS): {len(c1_diffs)}")
    print(f"C2 frames (level > -30 dBFS): {len(c2_diffs)}")

    if len(c1_diffs) > 10 and len(c2_diffs) > 10:
        c1_avg = np.mean(c1_diffs, axis=0)
        c2_avg = np.mean(c2_diffs, axis=0)

        # 关键频点增益
        test_freqs = [250, 500, 1000, 2000, 4000, 8000]

        print("\n" + "=" * 70)
        print("C1 状态 (安静段落) 频谱增益:")
        print("-" * 50)
        for f in test_freqs:
            idx = np.argmin(np.abs(freqs - f))
            gain = c1_avg[idx]
            print(f"  {f:5d} Hz: {gain:+.1f} dB")

        # 计算倾斜
        c1_250 = band_power_db(c1_avg, freqs, 200, 300)
        c1_4k = band_power_db(c1_avg, freqs, 3500, 4500)
        print(f"\n  倾斜 (4kHz - 250Hz): {c1_4k - c1_250:+.1f} dB")
        print(f"  理论值: -30 dB")

        print("\n" + "=" * 70)
        print("C2 状态 (响亮段落) 频谱增益:")
        print("-" * 50)
        for f in test_freqs:
            idx = np.argmin(np.abs(freqs - f))
            gain = c2_avg[idx]
            print(f"  {f:5d} Hz: {gain:+.1f} dB")

        c2_250 = band_power_db(c2_avg, freqs, 200, 300)
        c2_4k = band_power_db(c2_avg, freqs, 3500, 4500)
        print(f"\n  倾斜 (4kHz - 250Hz): {c2_4k - c2_250:+.1f} dB")
        print(f"  理论值: +30 dB")

        print("\n" + "=" * 70)
        print("总结:")
        print("-" * 50)
        print(f"  C1 实测倾斜: {c1_4k - c1_250:+.1f} dB (理论 -30 dB)")
        print(f"  C2 实测倾斜: {c2_4k - c2_250:+.1f} dB (理论 +30 dB)")
        print(f"  C1-C2 差异: {(c1_4k - c1_250) - (c2_4k - c2_250):.1f} dB (理论 -60 dB)")

if __name__ == "__main__":
    analyze_tilt_detail("D MNF.flac", "Tomatis_D_30m_declick.flac")
