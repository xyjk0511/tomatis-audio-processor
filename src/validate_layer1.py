#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer1 验证工具 - 不依赖基准录音的独立验证

验证目标说明:
  本工具验证的是"算法实现是否符合参数设定"，而非"与硬件一模一样"。
  不依赖基准录音时，你能验证的是:
    1. 门控逻辑是否按 RMS dBFS + 回差 + 延迟执行
    2. C1/C2 滤波形状是否按 fc/slope/gain 生效
    3. 工程完整性（长度、削波、无异常瞬态）

验证项目:
  A. 工程检查: 采样率/位深/声道、样点数一致、峰值安全
  B. Gate 独立复算: 从输入音频重新计算 state，与 csv 对照
  C. 条件频谱验证: C1/C2 的 Delta 曲线是否符合理论形状
  D. 判定: RMSE 容忍度 PASS/FAIL
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-12


# ============================================================
# 辅助函数
# ============================================================

def rms_dbfs(x_mono: np.ndarray) -> float:
    """计算单声道帧的 RMS dBFS"""
    r = np.sqrt(np.mean(x_mono * x_mono) + EPS)
    return float(20.0 * np.log10(r + EPS))


def build_tilt_gain_db(freqs, fc, slope_db_per_oct, low_gain_db, high_gain_db):
    """生成理论倾斜增益曲线 (dB)"""
    f = np.maximum(freqs, 1.0)
    x = np.log2(f / fc).astype(np.float32)
    g = np.zeros_like(x, dtype=np.float32)

    d_low = slope_db_per_oct * np.maximum(0.0, -x)
    g_low = np.sign(low_gain_db) * np.minimum(d_low, abs(low_gain_db))
    g[x < 0] = g_low[x < 0]

    d_hi = slope_db_per_oct * np.maximum(0.0, x)
    g_hi = np.sign(high_gain_db) * np.minimum(d_hi, abs(high_gain_db))
    g[x > 0] = g_hi[x > 0]

    return g


# ============================================================
# A. 工程检查
# ============================================================

def check_engineering(in_path, out_path):
    """检查采样率/声道/样点数/峰值"""
    results = {}

    info_in = sf.info(in_path)
    info_out = sf.info(out_path)

    results['sr_in'] = info_in.samplerate
    results['sr_out'] = info_out.samplerate
    results['sr_match'] = info_in.samplerate == info_out.samplerate

    results['ch_in'] = info_in.channels
    results['ch_out'] = info_out.channels
    results['ch_match'] = info_in.channels == info_out.channels

    results['frames_in'] = info_in.frames
    results['frames_out'] = info_out.frames
    results['frames_match'] = info_in.frames == info_out.frames
    results['frames_diff'] = info_out.frames - info_in.frames

    # 峰值检查
    y, _ = sf.read(out_path, dtype='float32')
    peak = np.max(np.abs(y))
    results['peak'] = peak
    results['peak_safe'] = peak < 0.98

    return results


# ============================================================
# B. Gate 独立复算
# ============================================================

def load_state_csv(csv_path):
    """加载 state_csv"""
    frames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'frame_idx': int(row['frame_idx']),
                'time_sec': float(row['time_sec']),
                'level_dbfs': float(row['level_dbfs']),
                'state': row['state']
            })
    return frames


def simulate_gate(x, sr, n_fft, hop, threshold_dbfs, hyst_db, up_delay_ms):
    """独立复算 Gate 状态"""
    Ton = threshold_dbfs + hyst_db / 2
    Toff = threshold_dbfs - hyst_db / 2
    up_delay_samples = int(up_delay_ms * sr / 1000)

    total_frames = len(x)
    pad_len = n_fft // 2

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ch = x.shape[1]

    x_pad = np.vstack([np.zeros((pad_len, ch), dtype=x.dtype), x, np.zeros((pad_len, ch), dtype=x.dtype)])

    state = 1
    pending_c2_at = None
    states = []
    levels = []

    next_start = 0
    frame_idx = 0

    while next_start + n_fft <= len(x_pad):
        frame = x_pad[next_start:next_start + n_fft, :]
        mono = np.sqrt(np.mean(frame**2, axis=1))
        level = rms_dbfs(mono)

        # 只记录落在原音频范围内的帧
        orig_start = next_start - pad_len
        if 0 <= orig_start < total_frames:
            # Gate 状态机
            old_state = state
            if state == 1:
                if level >= Ton:
                    if pending_c2_at is None:
                        pending_c2_at = next_start + up_delay_samples
                else:
                    pending_c2_at = None
                if pending_c2_at is not None and next_start >= pending_c2_at:
                    state = 2
                    pending_c2_at = None
            else:
                if level <= Toff:
                    state = 1
                    pending_c2_at = None

            states.append('C1' if state == 1 else 'C2')
            levels.append(level)
            frame_idx += 1

        next_start += hop

    return states, levels


def compare_gate_states(csv_states, sim_states, sim_levels, csv_levels):
    """对比 CSV 状态与独立复算状态"""
    n = min(len(csv_states), len(sim_states))

    mismatch_count = 0
    level_diffs = []

    for i in range(n):
        if csv_states[i] != sim_states[i]:
            mismatch_count += 1
        level_diffs.append(abs(csv_levels[i] - sim_levels[i]))

    mismatch_rate = mismatch_count / n if n > 0 else 0

    # 切换次数
    csv_switches = sum(1 for i in range(1, len(csv_states)) if csv_states[i] != csv_states[i-1])
    sim_switches = sum(1 for i in range(1, len(sim_states)) if sim_states[i] != sim_states[i-1])

    return {
        'total_frames': n,
        'mismatch_count': mismatch_count,
        'mismatch_rate': mismatch_rate,
        'csv_switches': csv_switches,
        'sim_switches': sim_switches,
        'switch_diff': abs(csv_switches - sim_switches),
        'level_max_diff': max(level_diffs) if level_diffs else 0,
        'level_mean_diff': np.mean(level_diffs) if level_diffs else 0
    }


# ============================================================
# C. Gate 统计
# ============================================================

def analyze_gate_stats(states):
    """分析 Gate 统计: C2占比、切换次数、run length"""
    n = len(states)
    if n == 0:
        return {}

    c2_count = sum(1 for s in states if s == 'C2')
    c2_ratio = c2_count / n

    # 切换次数
    switch_count = sum(1 for i in range(1, n) if states[i] != states[i-1])

    # Run length 分布
    run_lengths = []
    current_run = 1
    for i in range(1, n):
        if states[i] == states[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    # 短段统计 (抖动检测)
    short_runs = sum(1 for r in run_lengths if r <= 3)
    short_run_ratio = short_runs / len(run_lengths) if run_lengths else 0

    return {
        'total_frames': n,
        'c2_count': c2_count,
        'c2_ratio': c2_ratio,
        'switch_count': switch_count,
        'run_count': len(run_lengths),
        'run_min': min(run_lengths) if run_lengths else 0,
        'run_max': max(run_lengths) if run_lengths else 0,
        'run_median': np.median(run_lengths) if run_lengths else 0,
        'short_runs': short_runs,
        'short_run_ratio': short_run_ratio
    }


# ============================================================
# D. 条件频谱验证
# ============================================================

def find_stable_frames(states, margin=2):
    """找到稳定帧 (前后 margin 帧状态相同)"""
    n = len(states)
    c1_stable = []
    c2_stable = []

    for i in range(margin, n - margin):
        window = states[i - margin:i + margin + 1]
        if all(s == 'C1' for s in window):
            c1_stable.append(i)
        elif all(s == 'C2' for s in window):
            c2_stable.append(i)

    return c1_stable, c2_stable


def compute_conditional_spectrum(x, y, sr, states, n_fft, hop, level_threshold=-60):
    """
    计算条件频谱 Delta(f) = 20*log10(|Y|/|X|)

    参数:
        x: 输入音频
        y: 输出音频
        sr: 采样率
        states: 状态列表
        n_fft: FFT 窗长
        hop: 跳步
        level_threshold: 电平阈值 (dBFS)，低于此值的帧不纳入
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    ch = x.shape[1]
    pad_len = n_fft // 2

    x_pad = np.vstack([np.zeros((pad_len, ch), dtype=x.dtype), x, np.zeros((pad_len, ch), dtype=x.dtype)])
    y_pad = np.vstack([np.zeros((pad_len, ch), dtype=y.dtype), y, np.zeros((pad_len, ch), dtype=y.dtype)])

    # 找稳定帧
    c1_stable, c2_stable = find_stable_frames(states, margin=2)

    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    n_bins = len(freqs)

    c1_ratios = []
    c2_ratios = []

    total_frames = len(x)
    win = np.hanning(n_fft).astype(np.float32)

    for frame_idx, stable_list, ratio_list in [(c1_stable, c1_stable, c1_ratios),
                                                 (c2_stable, c2_stable, c2_ratios)]:
        for idx in stable_list:
            # 计算帧位置
            orig_start = idx * hop
            if orig_start < 0 or orig_start + n_fft > len(x):
                continue

            start = orig_start + pad_len

            # 检查电平
            frame_x = x_pad[start:start + n_fft, :]
            mono = np.sqrt(np.mean(frame_x**2, axis=1))
            level = rms_dbfs(mono)
            if level < level_threshold:
                continue

            frame_y = y_pad[start:start + n_fft, :]

            # 计算频谱 (平均声道)
            X = np.zeros(n_bins, dtype=np.float32)
            Y = np.zeros(n_bins, dtype=np.float32)
            for c in range(ch):
                X += np.abs(np.fft.rfft(frame_x[:, c] * win))
                Y += np.abs(np.fft.rfft(frame_y[:, c] * win))
            X /= ch
            Y /= ch

            # 避免除零
            X = np.maximum(X, 1e-10)
            ratio = Y / X
            ratio_list.append(ratio)

    # 稳定帧列表修正 (前面循环有错误，重新实现)
    c1_ratios = []
    c2_ratios = []

    for idx in c1_stable:
        orig_start = idx * hop
        if orig_start < 0 or orig_start + n_fft > len(x):
            continue
        start = orig_start + pad_len
        frame_x = x_pad[start:start + n_fft, :]
        mono = np.sqrt(np.mean(frame_x**2, axis=1))
        level = rms_dbfs(mono)
        if level < level_threshold:
            continue
        frame_y = y_pad[start:start + n_fft, :]
        X = np.zeros(n_bins, dtype=np.float32)
        Y = np.zeros(n_bins, dtype=np.float32)
        for c in range(ch):
            X += np.abs(np.fft.rfft(frame_x[:, c] * win))
            Y += np.abs(np.fft.rfft(frame_y[:, c] * win))
        X /= ch
        Y /= ch
        X = np.maximum(X, 1e-10)
        c1_ratios.append(Y / X)

    for idx in c2_stable:
        orig_start = idx * hop
        if orig_start < 0 or orig_start + n_fft > len(x):
            continue
        start = orig_start + pad_len
        frame_x = x_pad[start:start + n_fft, :]
        mono = np.sqrt(np.mean(frame_x**2, axis=1))
        level = rms_dbfs(mono)
        if level < level_threshold:
            continue
        frame_y = y_pad[start:start + n_fft, :]
        X = np.zeros(n_bins, dtype=np.float32)
        Y = np.zeros(n_bins, dtype=np.float32)
        for c in range(ch):
            X += np.abs(np.fft.rfft(frame_x[:, c] * win))
            Y += np.abs(np.fft.rfft(frame_y[:, c] * win))
        X /= ch
        Y /= ch
        X = np.maximum(X, 1e-10)
        c2_ratios.append(Y / X)

    # 中位数统计
    if c1_ratios:
        c1_median = np.median(np.array(c1_ratios), axis=0)
        c1_db = 20 * np.log10(c1_median + EPS)
    else:
        c1_db = np.zeros(n_bins)

    if c2_ratios:
        c2_median = np.median(np.array(c2_ratios), axis=0)
        c2_db = 20 * np.log10(c2_median + EPS)
    else:
        c2_db = np.zeros(n_bins)

    return freqs, c1_db, c2_db, len(c1_ratios), len(c2_ratios)


def compute_spectrum_rmse(measured_db, theory_db, freqs, f_low, f_high):
    """计算指定频段的 RMSE"""
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    diff = measured_db[mask] - theory_db[mask]
    return np.sqrt(np.mean(diff**2))


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Layer1 验证工具')
    parser.add_argument('-i', '--input', required=True, help='原始输入音频')
    parser.add_argument('-o', '--output', required=True, help='Layer1 输出音频')
    parser.add_argument('--state_csv', required=True, help='状态 CSV 文件')

    # Gate 参数
    parser.add_argument('--gate_ui', type=float, default=50)
    parser.add_argument('--gate_scale', type=float, default=1.0)
    parser.add_argument('--gate_offset', type=float, default=-61.08)
    parser.add_argument('--hyst_db', type=float, default=1.0)
    parser.add_argument('--up_delay_ms', type=float, default=0)

    # 滤波器参数
    parser.add_argument('--fc', type=float, default=1000)
    parser.add_argument('--slope', type=float, default=12)
    parser.add_argument('--c1_low', type=float, default=5.0)
    parser.add_argument('--c1_high', type=float, default=-5.0)
    parser.add_argument('--c2_low', type=float, default=-5.0)
    parser.add_argument('--c2_high', type=float, default=5.0)

    # STFT 参数
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--hop', type=int, default=2048)

    # 输出
    parser.add_argument('--out_csv', default='layer1_spectrum_check.csv')
    parser.add_argument('--out_png', default='layer1_spectrum_check.png')

    args = parser.parse_args()

    print('=' * 60)
    print('Layer1 验证工具')
    print('=' * 60)
    print()
    print('验证目标: 算法实现是否符合参数设定')
    print('  - 门控逻辑: RMS dBFS + 回差 + 延迟')
    print('  - 滤波形状: fc/slope/gain')
    print('  - 工程完整性: 长度、削波')
    print('注意: 本工具不验证"与硬件一模一样"')
    print()

    threshold_dbfs = args.gate_scale * args.gate_ui + args.gate_offset
    print(f'参数:')
    print(f'  Gate: UI={args.gate_ui}, T={threshold_dbfs:.2f} dBFS, hyst={args.hyst_db} dB, delay={args.up_delay_ms} ms')
    print(f'  Filter: fc={args.fc} Hz, slope={args.slope} dB/oct')
    print(f'  C1: low={args.c1_low} dB, high={args.c1_high} dB')
    print(f'  C2: low={args.c2_low} dB, high={args.c2_high} dB')
    print()

    results = {'pass': True, 'checks': {}}

    # ========================================
    # A. 工程检查
    # ========================================
    print('-' * 40)
    print('A. 工程检查')
    print('-' * 40)

    eng = check_engineering(args.input, args.output)

    print(f'  采样率: {eng["sr_in"]} -> {eng["sr_out"]} {"PASS" if eng["sr_match"] else "FAIL"}')
    print(f'  声道数: {eng["ch_in"]} -> {eng["ch_out"]} {"PASS" if eng["ch_match"] else "FAIL"}')
    print(f'  样点数: {eng["frames_in"]} -> {eng["frames_out"]} (diff={eng["frames_diff"]}) {"PASS" if eng["frames_match"] else "FAIL"}')
    print(f'  峰值: {eng["peak"]:.4f} {"PASS" if eng["peak_safe"] else "FAIL (>=0.98)"}')

    results['checks']['engineering'] = {
        'sr_match': eng['sr_match'],
        'ch_match': eng['ch_match'],
        'frames_match': eng['frames_match'],
        'peak_safe': eng['peak_safe']
    }
    if not all([eng['sr_match'], eng['ch_match'], eng['frames_match'], eng['peak_safe']]):
        results['pass'] = False

    # ========================================
    # B. Gate 独立复算
    # ========================================
    print()
    print('-' * 40)
    print('B. Gate 独立复算')
    print('-' * 40)

    csv_frames = load_state_csv(args.state_csv)
    csv_states = [f['state'] for f in csv_frames]
    csv_levels = [f['level_dbfs'] for f in csv_frames]

    x, sr = sf.read(args.input, dtype='float32')
    sim_states, sim_levels = simulate_gate(x, sr, args.n_fft, args.hop,
                                            threshold_dbfs, args.hyst_db, args.up_delay_ms)

    cmp = compare_gate_states(csv_states, sim_states, sim_levels, csv_levels)

    print(f'  总帧数: {cmp["total_frames"]}')
    print(f'  状态不匹配: {cmp["mismatch_count"]} ({cmp["mismatch_rate"]*100:.2f}%)')
    print(f'  切换次数: CSV={cmp["csv_switches"]}, SIM={cmp["sim_switches"]}, diff={cmp["switch_diff"]}')
    print(f'  电平最大差: {cmp["level_max_diff"]:.4f} dB')
    print(f'  电平平均差: {cmp["level_mean_diff"]:.4f} dB')

    gate_ok = cmp['mismatch_rate'] < 0.01 and cmp['level_max_diff'] < 0.1
    print(f'  结果: {"PASS" if gate_ok else "FAIL (mismatch>1% or level_diff>0.1dB)"}')

    results['checks']['gate_verify'] = {
        'mismatch_rate': cmp['mismatch_rate'],
        'level_max_diff': cmp['level_max_diff'],
        'pass': gate_ok
    }
    if not gate_ok:
        results['pass'] = False

    # ========================================
    # C. Gate 统计
    # ========================================
    print()
    print('-' * 40)
    print('C. Gate 统计')
    print('-' * 40)

    stats = analyze_gate_stats(csv_states)

    duration_min = stats['total_frames'] * args.hop / sr / 60
    switches_per_min = stats['switch_count'] / duration_min if duration_min > 0 else 0

    print(f'  C2 占比: {stats["c2_ratio"]*100:.1f}%')
    print(f'  切换次数: {stats["switch_count"]} (约 {switches_per_min:.1f}/min)')
    print(f'  Run length: min={stats["run_min"]}, max={stats["run_max"]}, median={stats["run_median"]:.0f}')
    print(f'  短段(<=3帧): {stats["short_runs"]} ({stats["short_run_ratio"]*100:.1f}%)')

    # 判定
    c2_ratio_ok = 0.05 <= stats['c2_ratio'] <= 0.95
    jitter_ok = stats['short_run_ratio'] < 0.3

    print(f'  C2占比范围: {"PASS" if c2_ratio_ok else "WARN (极端值)"}')
    print(f'  抖动检测: {"PASS" if jitter_ok else "WARN (短段过多)"}')

    results['checks']['gate_stats'] = {
        'c2_ratio': stats['c2_ratio'],
        'short_run_ratio': stats['short_run_ratio'],
        'c2_ratio_ok': c2_ratio_ok,
        'jitter_ok': jitter_ok
    }

    # ========================================
    # D. 条件频谱验证
    # ========================================
    print()
    print('-' * 40)
    print('D. 条件频谱验证')
    print('-' * 40)

    y, _ = sf.read(args.output, dtype='float32')

    freqs, c1_db, c2_db, c1_n, c2_n = compute_conditional_spectrum(
        x, y, sr, csv_states, args.n_fft, args.hop, level_threshold=-60
    )

    print(f'  稳定帧: C1={c1_n}, C2={c2_n}')

    # 理论曲线
    c1_theory = build_tilt_gain_db(freqs, args.fc, args.slope, args.c1_low, args.c1_high)
    c2_theory = build_tilt_gain_db(freqs, args.fc, args.slope, args.c2_low, args.c2_high)

    # RMSE 计算
    bands = [
        ('low', 100, 800),
        ('mid', 800, 1200),
        ('high', 2000, 8000)
    ]

    print(f'  C1 RMSE:')
    c1_rmse_all = []
    for name, f_low, f_high in bands:
        rmse = compute_spectrum_rmse(c1_db, c1_theory, freqs, f_low, f_high)
        c1_rmse_all.append(rmse)
        print(f'    {name} ({f_low}-{f_high}Hz): {rmse:.2f} dB')

    print(f'  C2 RMSE:')
    c2_rmse_all = []
    for name, f_low, f_high in bands:
        rmse = compute_spectrum_rmse(c2_db, c2_theory, freqs, f_low, f_high)
        c2_rmse_all.append(rmse)
        print(f'    {name} ({f_low}-{f_high}Hz): {rmse:.2f} dB')

    # 判定: RMSE < 1.5 dB
    spectrum_ok = max(c1_rmse_all + c2_rmse_all) < 1.5
    print(f'  结果: {"PASS" if spectrum_ok else "FAIL (RMSE >= 1.5 dB)"}')

    results['checks']['spectrum'] = {
        'c1_rmse': c1_rmse_all,
        'c2_rmse': c2_rmse_all,
        'pass': spectrum_ok
    }
    if not spectrum_ok:
        results['pass'] = False

    # ========================================
    # 输出 CSV
    # ========================================
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['freq_hz', 'c1_measured_db', 'c1_theory_db', 'c2_measured_db', 'c2_theory_db'])
        for i, freq in enumerate(freqs):
            writer.writerow([f'{freq:.2f}', f'{c1_db[i]:.4f}', f'{c1_theory[i]:.4f}',
                           f'{c2_db[i]:.4f}', f'{c2_theory[i]:.4f}'])
    print(f'\n频谱数据已保存: {args.out_csv}')

    # ========================================
    # 绘图
    # ========================================
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # C1
        ax = axes[0]
        ax.semilogx(freqs, c1_db, 'b-', label='C1 measured', alpha=0.7)
        ax.semilogx(freqs, c1_theory, 'b--', label='C1 theory', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':')
        ax.axvline(args.fc, color='red', linestyle=':', label=f'fc={args.fc}Hz')
        ax.set_xlim(20, 20000)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'C1 Spectrum (n={c1_n})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # C2
        ax = axes[1]
        ax.semilogx(freqs, c2_db, 'r-', label='C2 measured', alpha=0.7)
        ax.semilogx(freqs, c2_theory, 'r--', label='C2 theory', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':')
        ax.axvline(args.fc, color='red', linestyle=':', label=f'fc={args.fc}Hz')
        ax.set_xlim(20, 20000)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'C2 Spectrum (n={c2_n})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        print(f'频谱图已保存: {args.out_png}')
        plt.close()
    except ImportError:
        print('matplotlib 未安装，跳过绘图')

    # ========================================
    # 最终判定
    # ========================================
    print()
    print('=' * 60)
    print('最终判定')
    print('=' * 60)

    all_checks = [
        ('工程检查', all(results['checks']['engineering'].values())),
        ('Gate复算', results['checks']['gate_verify']['pass']),
        ('条件频谱', results['checks']['spectrum']['pass'])
    ]

    for name, passed in all_checks:
        print(f'  {name}: {"PASS" if passed else "FAIL"}')

    print()
    if results['pass']:
        print('Layer1 验证: PASS')
        print('算法实现符合参数设定')
    else:
        print('Layer1 验证: FAIL')
        print('请检查上述 FAIL 项')

    return 0 if results['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
