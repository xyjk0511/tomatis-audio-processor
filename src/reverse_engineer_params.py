"""
反推 Tomatis 设备处理参数

通过比较输入和输出音频，反推设备的实际处理参数：
- Gate 阈值 (dBFS)
- C1/C2 状态分布
- 倾斜滤波器参数验证

方法：
1. 互相关对齐输入输出
2. 逐帧计算频谱差异
3. 用倾斜指数(高频增益-低频增益)判断C1/C2状态
4. 统计不同输入电平下的状态分布
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve
import csv

EPS = 1e-12
SR = 48000
N_FFT = 4096
HOP = 2048

def power_mono(x_lr):
    """双声道转能量单声道"""
    p = 0.5 * (x_lr[:,0]**2 + x_lr[:,1]**2)
    return np.sqrt(p + EPS)

def rms_dbfs(x):
    """计算RMS dBFS"""
    return 20 * np.log10(np.sqrt(np.mean(x*x) + EPS) + EPS)

def find_delay_xcorr(base_mono, cand_mono, sr=SR, ds_sr=2000):
    """互相关找延迟"""
    up, down = ds_sr, sr
    b = resample_poly(base_mono - base_mono.mean(), up, down).astype(np.float32)
    c = resample_poly(cand_mono - cand_mono.mean(), up, down).astype(np.float32)

    corr = fftconvolve(c, b[::-1], mode="full")
    k = int(np.argmax(corr))
    shift_ds = k - (len(b) - 1)
    delay_samples = int(round(shift_ds * (sr / ds_sr)))
    return delay_samples

def align_audio(base_lr, cand_lr, delay):
    """对齐两个音频"""
    if delay > 0:
        cand_lr = cand_lr[delay:]
    elif delay < 0:
        base_lr = base_lr[-delay:]
    n = min(len(base_lr), len(cand_lr))
    return base_lr[:n], cand_lr[:n]

def compute_frame_spectrum(frame, win):
    """计算单帧频谱(dB)"""
    X = np.fft.rfft(frame * win)
    mag = np.abs(X) + EPS
    return 20 * np.log10(mag)

def compute_tilt_index(spec_db, freqs, fc=1000):
    """
    计算倾斜指数: 高频平均增益 - 低频平均增益
    - 正值 = C2特征 (高频增强)
    - 负值 = C1特征 (低频增强)
    """
    # 低频: 200-500 Hz
    low_mask = (freqs >= 200) & (freqs < 500)
    # 高频: 2000-6000 Hz
    high_mask = (freqs >= 2000) & (freqs < 6000)

    low_avg = np.mean(spec_db[low_mask])
    high_avg = np.mean(spec_db[high_mask])

    return high_avg - low_avg

def analyze_device_params(input_path, output_path, out_csv=None):
    """主分析函数"""
    print("=" * 70)
    print("Tomatis 设备参数反推分析")
    print("=" * 70)

    # 读取文件
    print(f"\n输入文件: {input_path}")
    print(f"输出文件: {output_path}")

    inp_lr, sr1 = sf.read(input_path, dtype="float32", always_2d=True)
    out_lr, sr2 = sf.read(output_path, dtype="float32", always_2d=True)

    print(f"\n输入长度: {len(inp_lr)} samples ({len(inp_lr)/SR:.2f} sec)")
    print(f"输出长度: {len(out_lr)} samples ({len(out_lr)/SR:.2f} sec)")

    # 转单声道
    inp_mono = power_mono(inp_lr)
    out_mono = power_mono(out_lr)

    # 互相关对齐
    print("\n正在计算对齐延迟...")
    delay = find_delay_xcorr(inp_mono, out_mono)
    print(f"延迟: {delay} samples ({delay/SR*1000:.2f} ms)")

    # 对齐
    inp_aligned, out_aligned = align_audio(inp_lr, out_lr, delay)
    print(f"对齐后长度: {len(inp_aligned)} samples ({len(inp_aligned)/SR:.2f} sec)")

    # 准备STFT
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1/SR)
    n_frames = 1 + (len(inp_aligned) - N_FFT) // HOP

    print(f"\n分析帧数: {n_frames}")
    print("\n正在逐帧分析...")

    # 存储每帧数据
    frame_data = []

    for i in range(n_frames):
        st = i * HOP

        # 输入帧
        inp_frame = power_mono(inp_aligned[st:st+N_FFT])
        inp_level = rms_dbfs(inp_frame)

        # 输出帧
        out_frame = power_mono(out_aligned[st:st+N_FFT])

        # 计算频谱
        inp_spec = compute_frame_spectrum(inp_frame, win)
        out_spec = compute_frame_spectrum(out_frame, win)

        # 频谱差 (输出 - 输入)
        diff_spec = out_spec - inp_spec

        # 倾斜指数
        tilt = compute_tilt_index(diff_spec, freqs)

        frame_data.append({
            'frame': i,
            'time': st / SR,
            'inp_level': inp_level,
            'tilt': tilt
        })

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1}/{n_frames} 帧...")

    print("\n分析完成!")

    # 统计分析
    print("\n" + "=" * 70)
    print("统计分析结果")
    print("=" * 70)

    # 按输入电平分组统计倾斜指数
    level_bins = [(-70, -60), (-60, -55), (-55, -50), (-50, -45),
                  (-45, -40), (-40, -35), (-35, -30), (-30, -25),
                  (-25, -20), (-20, -15), (-15, -10)]

    print("\n按输入电平分组的倾斜指数统计:")
    print("-" * 60)
    print(f"{'电平范围':<15} {'平均倾斜':<12} {'标准差':<10} {'帧数':<8} {'状态'}")
    print("-" * 60)

    for lo, hi in level_bins:
        frames_in_bin = [f for f in frame_data if lo <= f['inp_level'] < hi]
        if len(frames_in_bin) > 0:
            tilts = [f['tilt'] for f in frames_in_bin]
            avg_tilt = np.mean(tilts)
            std_tilt = np.std(tilts)
            # 判断状态: 负倾斜=C1, 正倾斜=C2
            state = "C1" if avg_tilt < 0 else "C2"
            print(f"{lo:>3}~{hi:<3} dBFS   {avg_tilt:>+8.1f} dB   {std_tilt:>6.1f}    {len(frames_in_bin):<6}   {state}")

    # 找Gate阈值
    print("\n" + "-" * 60)
    print("Gate 阈值估计:")

    # 找倾斜指数从负变正的转折点
    c1_frames = [f for f in frame_data if f['tilt'] < -5]
    c2_frames = [f for f in frame_data if f['tilt'] > 5]

    if len(c1_frames) > 0 and len(c2_frames) > 0:
        c1_levels = [f['inp_level'] for f in c1_frames]
        c2_levels = [f['inp_level'] for f in c2_frames]

        c1_max = max(c1_levels)
        c2_min = min(c2_levels)

        print(f"  C1 帧数: {len(c1_frames)} (倾斜 < -5dB)")
        print(f"  C2 帧数: {len(c2_frames)} (倾斜 > +5dB)")
        print(f"  C1 最大电平: {c1_max:.1f} dBFS")
        print(f"  C2 最小电平: {c2_min:.1f} dBFS")
        print(f"  估计 Gate 阈值: {(c1_max + c2_min)/2:.1f} dBFS")
    else:
        print("  无法估计Gate阈值 - 未检测到明显的C1/C2分离")
        print(f"  C1 帧数 (倾斜<-5dB): {len(c1_frames)}")
        print(f"  C2 帧数 (倾斜>+5dB): {len(c2_frames)}")

    # 倾斜指数直方图
    all_tilts = [f['tilt'] for f in frame_data]
    print("\n倾斜指数分布:")
    hist_bins = [(-40, -30), (-30, -20), (-20, -10), (-10, 0),
                 (0, 10), (10, 20), (20, 30), (30, 40)]
    for lo, hi in hist_bins:
        count = sum(1 for t in all_tilts if lo <= t < hi)
        pct = count / len(all_tilts) * 100
        bar = "#" * int(pct / 2)
        print(f"  {lo:>+3}~{hi:>+3} dB: {count:>5} ({pct:>5.1f}%) {bar}")

    # 保存CSV
    if out_csv:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'time_sec', 'inp_level_dbfs', 'tilt_db'])
            for fd in frame_data:
                writer.writerow([fd['frame'], f"{fd['time']:.3f}",
                               f"{fd['inp_level']:.2f}", f"{fd['tilt']:.2f}"])
        print(f"\n详细数据已保存: {out_csv}")

    return frame_data

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="反推Tomatis设备参数")
    ap.add_argument("-i", "--input", required=True, help="原始输入文件")
    ap.add_argument("-o", "--output", required=True, help="设备输出文件")
    ap.add_argument("--csv", default=None, help="输出CSV文件路径")
    args = ap.parse_args()

    analyze_device_params(args.input, args.output, args.csv)
