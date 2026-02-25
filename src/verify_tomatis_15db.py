#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tomatis ±15dB 完整验证工具

验证 D_MNF_matched_15dB.flac 是否符合设定的 Tomatis 规则，并量化效果强度。

验证项目:
  A. 工程检查: 采样率/位深/声道、样点数一致、峰值安全、DC偏移
  B. Gate 统计: C2占比、切换次数、run length、抖动检测、自洽性验证
  C. 条件频谱验证: C1/C2 的 Delta 曲线是否符合 ±15dB 理论形状
  D. 效果量化: Tilt Index (高低频能量比) 分析

作者: DSP 分析工具
日期: 2026-01-20
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
    """
    生成理论倾斜增益曲线 (dB)
    
    关键频率:
    - 低频封顶: f_lo = fc * 2^(-|G_lo| / slope)  ~420 Hz for ±15dB
    - 高频封顶: f_hi = fc * 2^(|G_hi| / slope)   ~2380 Hz for ±15dB
    """
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
    """检查采样率/声道/样点数/峰值/DC偏移"""
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

    # 读取输出音频用于峰值和DC偏移检查
    y, _ = sf.read(out_path, dtype='float32')
    
    # 峰值检查
    peak = np.max(np.abs(y))
    results['peak'] = peak
    results['peak_safe'] = peak < 0.98
    results['peak_dbfs'] = 20 * np.log10(peak + EPS)

    # DC偏移检查
    dc_mean = np.mean(y)
    results['dc_mean'] = dc_mean
    results['dc_safe'] = abs(dc_mean) < 0.001

    return results


# ============================================================
# B. Gate 模拟与统计
# ============================================================

def simulate_gate(x, sr, n_fft, hop, threshold_dbfs, hyst_db, up_delay_ms):
    """独立模拟 Gate 状态（从输入音频计算）"""
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
    times = []

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
            times.append(orig_start / sr)
            frame_idx += 1

        next_start += hop

    return states, levels, times


def analyze_gate_stats(states, levels, sr, hop):
    """分析 Gate 统计: C2占比、切换次数、run length、抖动检测"""
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

    # 计算时长和切换频率
    duration_sec = n * hop / sr
    duration_min = duration_sec / 60
    switches_per_min = switch_count / duration_min if duration_min > 0 else 0

    # RMS 与状态关联分析
    c1_levels = [levels[i] for i in range(n) if states[i] == 'C1']
    c2_levels = [levels[i] for i in range(n) if states[i] == 'C2']

    return {
        'total_frames': n,
        'duration_sec': duration_sec,
        'duration_min': duration_min,
        'c2_count': c2_count,
        'c2_ratio': c2_ratio,
        'switch_count': switch_count,
        'switches_per_min': switches_per_min,
        'run_count': len(run_lengths),
        'run_min': min(run_lengths) if run_lengths else 0,
        'run_max': max(run_lengths) if run_lengths else 0,
        'run_median': np.median(run_lengths) if run_lengths else 0,
        'run_mean': np.mean(run_lengths) if run_lengths else 0,
        'short_runs': short_runs,
        'short_run_ratio': short_run_ratio,
        'c1_level_mean': np.mean(c1_levels) if c1_levels else 0,
        'c2_level_mean': np.mean(c2_levels) if c2_levels else 0,
    }


# ============================================================
# C. 条件频谱验证
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
    按 C1/C2 状态分组，使用中位数统计
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
    win = np.hanning(n_fft).astype(np.float32)

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


def compute_spectrum_metrics(freqs, c1_db, c2_db, c1_theory, c2_theory, fc, gain_limit):
    """计算频谱验收指标"""
    metrics = {}
    
    # 1. RMSE 计算 (100-8000Hz)
    analysis_mask = (freqs >= 100) & (freqs <= 8000)
    if np.any(analysis_mask):
        c1_diff = c1_db[analysis_mask] - c1_theory[analysis_mask]
        c2_diff = c2_db[analysis_mask] - c2_theory[analysis_mask]
        metrics['c1_rmse'] = np.sqrt(np.mean(c1_diff**2))
        metrics['c2_rmse'] = np.sqrt(np.mean(c2_diff**2))
    
    # 2. fc (1000Hz) 过零误差
    fc_mask = (freqs >= 900) & (freqs <= 1100)
    if np.any(fc_mask):
        metrics['c1_fc_error'] = abs(np.mean(c1_db[fc_mask]))
        metrics['c2_fc_error'] = abs(np.mean(c2_db[fc_mask]))
    
    # 3. 低频平台误差 (目标 ±gain_limit dB)
    # 低频封顶约 ~420 Hz for ±15dB
    low_platform_mask = (freqs >= 100) & (freqs <= 350)
    if np.any(low_platform_mask):
        c1_low_mean = np.mean(c1_db[low_platform_mask])
        c2_low_mean = np.mean(c2_db[low_platform_mask])
        metrics['c1_low_platform'] = c1_low_mean
        metrics['c2_low_platform'] = c2_low_mean
        metrics['c1_low_platform_error'] = abs(c1_low_mean - gain_limit)
        metrics['c2_low_platform_error'] = abs(c2_low_mean - (-gain_limit))
    
    # 4. 高频平台误差 (目标 ±gain_limit dB)
    # 高频封顶约 ~2380 Hz for ±15dB
    high_platform_mask = (freqs >= 3000) & (freqs <= 8000)
    if np.any(high_platform_mask):
        c1_high_mean = np.mean(c1_db[high_platform_mask])
        c2_high_mean = np.mean(c2_db[high_platform_mask])
        metrics['c1_high_platform'] = c1_high_mean
        metrics['c2_high_platform'] = c2_high_mean
        metrics['c1_high_platform_error'] = abs(c1_high_mean - (-gain_limit))
        metrics['c2_high_platform_error'] = abs(c2_high_mean - gain_limit)
    
    return metrics


# ============================================================
# D. 效果量化 - Tilt Index
# ============================================================

def compute_tilt_index(x, y, sr, states, n_fft, hop):
    """
    计算 Tilt Index = 10*log10(E_hi/E_lo)
    - E_lo: 200-1000 Hz
    - E_hi: 2000-8000 Hz
    
    返回输入/输出/各状态的 TI 分布
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    ch = x.shape[1]
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    win = np.hanning(n_fft).astype(np.float32)
    
    # 频段 mask
    lo_mask = (freqs >= 200) & (freqs <= 1000)
    hi_mask = (freqs >= 2000) & (freqs <= 8000)
    
    ti_input = []
    ti_output = []
    ti_c1 = []
    ti_c2 = []
    
    n_frames = len(states)
    
    for i in range(n_frames):
        orig_start = i * hop
        if orig_start + n_fft > len(x):
            break
        
        # 输入帧
        frame_x = x[orig_start:orig_start + n_fft, :]
        X = np.zeros(len(freqs), dtype=np.float32)
        for c in range(ch):
            X += np.abs(np.fft.rfft(frame_x[:, c] * win))**2
        X /= ch
        
        E_lo_x = np.sum(X[lo_mask])
        E_hi_x = np.sum(X[hi_mask])
        if E_lo_x > EPS:
            ti_x = 10 * np.log10(E_hi_x / E_lo_x + EPS)
            ti_input.append(ti_x)
        
        # 输出帧
        frame_y = y[orig_start:orig_start + n_fft, :]
        Y = np.zeros(len(freqs), dtype=np.float32)
        for c in range(ch):
            Y += np.abs(np.fft.rfft(frame_y[:, c] * win))**2
        Y /= ch
        
        E_lo_y = np.sum(Y[lo_mask])
        E_hi_y = np.sum(Y[hi_mask])
        if E_lo_y > EPS:
            ti_y = 10 * np.log10(E_hi_y / E_lo_y + EPS)
            ti_output.append(ti_y)
            
            if states[i] == 'C1':
                ti_c1.append(ti_y)
            else:
                ti_c2.append(ti_y)
    
    return {
        'input': np.array(ti_input),
        'output': np.array(ti_output),
        'c1': np.array(ti_c1),
        'c2': np.array(ti_c2)
    }


def analyze_tilt_index(ti_data):
    """分析 Tilt Index 结果"""
    results = {}
    
    for key in ['input', 'output', 'c1', 'c2']:
        arr = ti_data[key]
        if len(arr) > 0:
            results[f'{key}_mean'] = np.mean(arr)
            results[f'{key}_std'] = np.std(arr)
            results[f'{key}_median'] = np.median(arr)
            results[f'{key}_min'] = np.min(arr)
            results[f'{key}_max'] = np.max(arr)
    
    # Tomatis 效果强度：C2 - C1 TI 差值
    if 'c1_mean' in results and 'c2_mean' in results:
        results['ti_effect'] = results['c2_mean'] - results['c1_mean']
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Tomatis ±15dB 完整验证工具')
    parser.add_argument('-i', '--input', required=True, help='原始输入音频')
    parser.add_argument('-o', '--output', required=True, help='处理后输出音频')

    # Gate 参数 (使用 calibration_v2.json 的默认值)
    parser.add_argument('--gate_ui', type=float, default=50)
    parser.add_argument('--gate_scale', type=float, default=1.0)
    parser.add_argument('--gate_offset', type=float, default=-61.08)
    parser.add_argument('--hyst_db', type=float, default=1.0)
    parser.add_argument('--up_delay_ms', type=float, default=0)

    # 滤波器参数 (±15dB 版本)
    parser.add_argument('--fc', type=float, default=1000)
    parser.add_argument('--slope', type=float, default=12)
    parser.add_argument('--c1_low', type=float, default=15.0)
    parser.add_argument('--c1_high', type=float, default=-15.0)
    parser.add_argument('--c2_low', type=float, default=-15.0)
    parser.add_argument('--c2_high', type=float, default=15.0)

    # STFT 参数
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--hop', type=int, default=2048)

    # 输出
    parser.add_argument('--out_prefix', default='verify_15db')

    args = parser.parse_args()

    print('=' * 70)
    print('Tomatis ±15dB 完整验证工具')
    print('=' * 70)
    print()
    
    gain_limit = abs(args.c1_low)  # 15 dB
    
    threshold_dbfs = args.gate_scale * args.gate_ui + args.gate_offset
    print(f'参数配置:')
    print(f'  Gate: UI={args.gate_ui}, T={threshold_dbfs:.2f} dBFS, hyst={args.hyst_db} dB')
    print(f'  Filter: fc={args.fc} Hz, slope={args.slope} dB/oct')
    print(f'  C1: low=+{args.c1_low} dB, high={args.c1_high} dB')
    print(f'  C2: low={args.c2_low} dB, high=+{args.c2_high} dB')
    print(f'  期望平台: 低频封顶 ~{args.fc * 2**(-gain_limit/args.slope):.0f} Hz, '
          f'高频封顶 ~{args.fc * 2**(gain_limit/args.slope):.0f} Hz')
    print()

    all_pass = True
    report_lines = []
    report_lines.append('Tomatis ±15dB 验证报告')
    report_lines.append('=' * 50)

    # ========================================
    # A. 工程检查
    # ========================================
    print('-' * 50)
    print('A. 工程检查')
    print('-' * 50)

    eng = check_engineering(args.input, args.output)

    sr_ok = eng['sr_match']
    ch_ok = eng['ch_match']
    frames_ok = eng['frames_match']
    peak_ok = eng['peak_safe']
    dc_ok = eng['dc_safe']

    print(f'  采样率: {eng["sr_in"]} -> {eng["sr_out"]} {"PASS" if sr_ok else "FAIL"}')
    print(f'  声道数: {eng["ch_in"]} -> {eng["ch_out"]} {"PASS" if ch_ok else "FAIL"}')
    print(f'  样点数: {eng["frames_in"]} -> {eng["frames_out"]} (diff={eng["frames_diff"]}) {"PASS" if frames_ok else "FAIL"}')
    print(f'  峰值: {eng["peak"]:.4f} ({eng["peak_dbfs"]:.2f} dBFS) {"PASS" if peak_ok else "FAIL (>=0.98)"}')
    print(f'  DC偏移: {eng["dc_mean"]:.6f} {"PASS" if dc_ok else "FAIL (>0.001)"}')

    eng_pass = sr_ok and ch_ok and frames_ok and peak_ok and dc_ok
    print(f'  工程检查结果: {"PASS" if eng_pass else "FAIL"}')
    
    if not eng_pass:
        all_pass = False
    
    report_lines.append('\nA. 工程检查')
    report_lines.append(f'  采样率一致: {"PASS" if sr_ok else "FAIL"}')
    report_lines.append(f'  声道一致: {"PASS" if ch_ok else "FAIL"}')
    report_lines.append(f'  样点一致: {"PASS" if frames_ok else "FAIL"} (diff={eng["frames_diff"]})')
    report_lines.append(f'  峰值安全: {"PASS" if peak_ok else "FAIL"} ({eng["peak"]:.4f})')
    report_lines.append(f'  DC偏移: {"PASS" if dc_ok else "FAIL"} ({eng["dc_mean"]:.6f})')

    # ========================================
    # B. Gate 统计
    # ========================================
    print()
    print('-' * 50)
    print('B. Gate 统计分析')
    print('-' * 50)

    print('  读取输入音频并模拟门控...')
    x, sr = sf.read(args.input, dtype='float32')
    y, _ = sf.read(args.output, dtype='float32')
    
    states, levels, times = simulate_gate(x, sr, args.n_fft, args.hop, 
                                          threshold_dbfs, args.hyst_db, args.up_delay_ms)
    
    stats = analyze_gate_stats(states, levels, sr, args.hop)

    print(f'  总帧数: {stats["total_frames"]}')
    print(f'  时长: {stats["duration_min"]:.2f} 分钟')
    print(f'  C2 占比: {stats["c2_ratio"]*100:.1f}%')
    print(f'  切换次数: {stats["switch_count"]} ({stats["switches_per_min"]:.1f}/min)')
    print(f'  Run length: min={stats["run_min"]}, max={stats["run_max"]}, median={stats["run_median"]:.0f}')
    print(f'  短段(<=3帧): {stats["short_runs"]} ({stats["short_run_ratio"]*100:.1f}%)')
    print(f'  C1平均电平: {stats["c1_level_mean"]:.2f} dBFS')
    print(f'  C2平均电平: {stats["c2_level_mean"]:.2f} dBFS')

    # 判定
    c2_ratio_ok = 0.05 <= stats['c2_ratio'] <= 0.95
    jitter_ok = stats['short_run_ratio'] < 0.3
    
    print(f'  C2占比范围(5%-95%): {"PASS" if c2_ratio_ok else "WARN"}')
    print(f'  抖动检测(<30%短段): {"PASS" if jitter_ok else "WARN"}')

    report_lines.append('\nB. Gate 统计')
    report_lines.append(f'  C2占比: {stats["c2_ratio"]*100:.1f}%')
    report_lines.append(f'  切换次数: {stats["switch_count"]} ({stats["switches_per_min"]:.1f}/min)')
    report_lines.append(f'  短段比例: {stats["short_run_ratio"]*100:.1f}%')

    # ========================================
    # C. 条件频谱验证
    # ========================================
    print()
    print('-' * 50)
    print('C. 条件频谱验证')
    print('-' * 50)

    freqs, c1_db, c2_db, c1_n, c2_n = compute_conditional_spectrum(
        x, y, sr, states, args.n_fft, args.hop, level_threshold=-60
    )

    print(f'  稳定帧: C1={c1_n}, C2={c2_n}')

    # 理论曲线
    c1_theory = build_tilt_gain_db(freqs, args.fc, args.slope, args.c1_low, args.c1_high)
    c2_theory = build_tilt_gain_db(freqs, args.fc, args.slope, args.c2_low, args.c2_high)

    # 计算指标
    metrics = compute_spectrum_metrics(freqs, c1_db, c2_db, c1_theory, c2_theory, args.fc, gain_limit)

    print(f'  C1 RMSE (100-8000Hz): {metrics.get("c1_rmse", 0):.2f} dB')
    print(f'  C2 RMSE (100-8000Hz): {metrics.get("c2_rmse", 0):.2f} dB')
    print(f'  C1 fc误差 (1000Hz): {metrics.get("c1_fc_error", 0):.2f} dB')
    print(f'  C2 fc误差 (1000Hz): {metrics.get("c2_fc_error", 0):.2f} dB')
    print(f'  C1 低频平台: {metrics.get("c1_low_platform", 0):.1f} dB (目标 +{gain_limit})')
    print(f'  C2 低频平台: {metrics.get("c2_low_platform", 0):.1f} dB (目标 -{gain_limit})')
    print(f'  C1 高频平台: {metrics.get("c1_high_platform", 0):.1f} dB (目标 -{gain_limit})')
    print(f'  C2 高频平台: {metrics.get("c2_high_platform", 0):.1f} dB (目标 +{gain_limit})')

    # 判定 (容差放宽到 0.5-1.5 dB)
    rmse_ok = metrics.get('c1_rmse', 99) < 1.5 and metrics.get('c2_rmse', 99) < 1.5
    fc_ok = metrics.get('c1_fc_error', 99) < 0.5 and metrics.get('c2_fc_error', 99) < 0.5
    platform_ok = (metrics.get('c1_low_platform_error', 99) < 3.0 and 
                   metrics.get('c2_low_platform_error', 99) < 3.0 and
                   metrics.get('c1_high_platform_error', 99) < 3.0 and 
                   metrics.get('c2_high_platform_error', 99) < 3.0)
    
    spectrum_pass = rmse_ok and fc_ok and platform_ok
    print(f'  RMSE验证(<1.5dB): {"PASS" if rmse_ok else "FAIL"}')
    print(f'  fc过零验证(<0.5dB): {"PASS" if fc_ok else "FAIL"}')
    print(f'  平台验证(<3dB误差): {"PASS" if platform_ok else "FAIL"}')
    print(f'  条件频谱结果: {"PASS" if spectrum_pass else "FAIL"}')

    if not spectrum_pass:
        all_pass = False

    report_lines.append('\nC. 条件频谱验证')
    report_lines.append(f'  C1 RMSE: {metrics.get("c1_rmse", 0):.2f} dB')
    report_lines.append(f'  C2 RMSE: {metrics.get("c2_rmse", 0):.2f} dB')
    report_lines.append(f'  fc误差: C1={metrics.get("c1_fc_error", 0):.2f}, C2={metrics.get("c2_fc_error", 0):.2f} dB')
    report_lines.append(f'  结果: {"PASS" if spectrum_pass else "FAIL"}')

    # 保存频谱 CSV
    csv_path = f'{args.out_prefix}_spectrum.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['freq_hz', 'c1_measured_db', 'c1_theory_db', 'c2_measured_db', 'c2_theory_db'])
        for i, freq in enumerate(freqs):
            writer.writerow([f'{freq:.2f}', f'{c1_db[i]:.4f}', f'{c1_theory[i]:.4f}',
                           f'{c2_db[i]:.4f}', f'{c2_theory[i]:.4f}'])
    print(f'  频谱数据已保存: {csv_path}')

    # ========================================
    # D. 效果量化 - Tilt Index
    # ========================================
    print()
    print('-' * 50)
    print('D. 效果量化 (Tilt Index)')
    print('-' * 50)

    ti_data = compute_tilt_index(x, y, sr, states, args.n_fft, args.hop)
    ti_results = analyze_tilt_index(ti_data)

    print(f'  输入 TI: mean={ti_results.get("input_mean", 0):.2f}, std={ti_results.get("input_std", 0):.2f}')
    print(f'  输出 TI: mean={ti_results.get("output_mean", 0):.2f}, std={ti_results.get("output_std", 0):.2f}')
    print(f'  C1 段 TI: mean={ti_results.get("c1_mean", 0):.2f}')
    print(f'  C2 段 TI: mean={ti_results.get("c2_mean", 0):.2f}')
    print(f'  Tomatis 效果强度 (C2-C1): {ti_results.get("ti_effect", 0):.2f} dB')

    # 效果判定: ±15dB 版本应该有明显的 TI 差异
    ti_effect = ti_results.get('ti_effect', 0)
    ti_effect_ok = ti_effect > 5.0  # 期望至少 5 dB 的效果差异
    print(f'  效果强度验证(>5dB): {"PASS" if ti_effect_ok else "WARN"}')

    report_lines.append('\nD. 效果量化')
    report_lines.append(f'  Tomatis效果强度(C2-C1): {ti_effect:.2f} dB')
    report_lines.append(f'  效果验证: {"PASS" if ti_effect_ok else "WARN (<5dB)"}')

    # ========================================
    # 绘图
    # ========================================
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # 1. 条件频谱图
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # C1
        ax = axes[0]
        ax.semilogx(freqs, c1_db, 'b-', label='C1 measured', alpha=0.7, linewidth=1.5)
        ax.semilogx(freqs, c1_theory, 'b--', label='C1 theory ±15dB', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(gain_limit, color='green', linestyle=':', alpha=0.5, label=f'+{gain_limit}dB platform')
        ax.axhline(-gain_limit, color='red', linestyle=':', alpha=0.5, label=f'-{gain_limit}dB platform')
        ax.axvline(args.fc, color='orange', linestyle='--', alpha=0.7, label=f'fc={args.fc}Hz')
        ax.axvline(args.fc * 2**(-gain_limit/args.slope), color='purple', linestyle=':', alpha=0.5)
        ax.axvline(args.fc * 2**(gain_limit/args.slope), color='purple', linestyle=':', alpha=0.5)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 20)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'C1 条件频谱 (稳定帧 n={c1_n}) - 低频增强/高频衰减')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # C2
        ax = axes[1]
        ax.semilogx(freqs, c2_db, 'r-', label='C2 measured', alpha=0.7, linewidth=1.5)
        ax.semilogx(freqs, c2_theory, 'r--', label='C2 theory ±15dB', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(gain_limit, color='green', linestyle=':', alpha=0.5, label=f'+{gain_limit}dB platform')
        ax.axhline(-gain_limit, color='red', linestyle=':', alpha=0.5, label=f'-{gain_limit}dB platform')
        ax.axvline(args.fc, color='orange', linestyle='--', alpha=0.7, label=f'fc={args.fc}Hz')
        ax.axvline(args.fc * 2**(-gain_limit/args.slope), color='purple', linestyle=':', alpha=0.5)
        ax.axvline(args.fc * 2**(gain_limit/args.slope), color='purple', linestyle=':', alpha=0.5)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 20)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'C2 条件频谱 (稳定帧 n={c2_n}) - 低频衰减/高频增强')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        spectrum_png = f'{args.out_prefix}_spectrum.png'
        plt.savefig(spectrum_png, dpi=150)
        print(f'  频谱图已保存: {spectrum_png}')
        plt.close()

        # 2. Tilt Index 分布图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 输入 vs 输出 TI 分布
        ax = axes[0]
        ax.hist(ti_data['input'], bins=50, alpha=0.5, label='Input', color='gray')
        ax.hist(ti_data['output'], bins=50, alpha=0.5, label='Output', color='blue')
        ax.axvline(ti_results.get('input_mean', 0), color='gray', linestyle='--', linewidth=2)
        ax.axvline(ti_results.get('output_mean', 0), color='blue', linestyle='--', linewidth=2)
        ax.set_xlabel('Tilt Index (dB)')
        ax.set_ylabel('Count')
        ax.set_title('Tilt Index 分布: 输入 vs 输出')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # C1 vs C2 TI 分布 (双峰特征)
        ax = axes[1]
        if len(ti_data['c1']) > 0:
            ax.hist(ti_data['c1'], bins=50, alpha=0.5, label=f'C1 (mean={ti_results.get("c1_mean", 0):.1f})', color='blue')
        if len(ti_data['c2']) > 0:
            ax.hist(ti_data['c2'], bins=50, alpha=0.5, label=f'C2 (mean={ti_results.get("c2_mean", 0):.1f})', color='red')
        ax.axvline(ti_results.get('c1_mean', 0), color='blue', linestyle='--', linewidth=2)
        ax.axvline(ti_results.get('c2_mean', 0), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Tilt Index (dB)')
        ax.set_ylabel('Count')
        ax.set_title(f'Tilt Index: C1 vs C2 (效果强度={ti_effect:.1f} dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        ti_png = f'{args.out_prefix}_tilt_index.png'
        plt.savefig(ti_png, dpi=150)
        print(f'  TI图已保存: {ti_png}')
        plt.close()

    except ImportError:
        print('  matplotlib 未安装，跳过绘图')

    # ========================================
    # 最终判定
    # ========================================
    print()
    print('=' * 70)
    print('最终判定')
    print('=' * 70)

    checks = [
        ('A. 工程检查', eng_pass),
        ('C. 条件频谱验证', spectrum_pass),
    ]

    for name, passed in checks:
        print(f'  {name}: {"PASS" if passed else "FAIL"}')

    print(f'  B. Gate统计: C2占比={stats["c2_ratio"]*100:.0f}%, 抖动={stats["short_run_ratio"]*100:.0f}%')
    print(f'  D. 效果量化: TI差值={ti_effect:.1f}dB')

    print()
    if all_pass:
        print('验证结果: PASS')
        print('D_MNF_matched_15dB.flac 符合设定的 Tomatis ±15dB 规则')
    else:
        print('验证结果: FAIL')
        print('请检查上述 FAIL 项')

    # 保存报告
    report_lines.append('\n' + '=' * 50)
    report_lines.append(f'总体结果: {"PASS" if all_pass else "FAIL"}')
    
    report_path = f'{args.out_prefix}_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\n综合报告已保存: {report_path}')

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
