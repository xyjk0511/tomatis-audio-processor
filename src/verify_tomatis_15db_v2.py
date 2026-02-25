#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tomatis ±15dB 自适应验证工具 v2

核心改进：
1. 门控阈值T自适应求解（目标C2=50%）
2. 条件频谱验证剔除弱能量帧
3. fc锚定用频带平均（900-1100Hz）而非单bin

纯数字音乐参数：
- hyst_db = 1.0 dB
- up_delay_ms = 0 ms  
- min_run_ms = 0 ms
- 目标 C2_ratio = 50%

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

    y, _ = sf.read(out_path, dtype='float32')
    
    peak = np.max(np.abs(y))
    results['peak'] = peak
    results['peak_safe'] = peak < 0.98
    results['peak_dbfs'] = 20 * np.log10(peak + EPS)

    dc_mean = np.mean(y)
    results['dc_mean'] = dc_mean
    results['dc_safe'] = abs(dc_mean) < 0.001

    return results


# ============================================================
# B. 自适应门控阈值求解
# ============================================================

def compute_frame_levels(x, sr, n_fft, hop):
    """计算每帧的 RMS dBFS 电平"""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ch = x.shape[1]
    
    pad_len = n_fft // 2
    x_pad = np.vstack([np.zeros((pad_len, ch), dtype=x.dtype), x, np.zeros((pad_len, ch), dtype=x.dtype)])
    
    levels = []
    next_start = 0
    total_frames = len(x)
    
    while next_start + n_fft <= len(x_pad):
        orig_start = next_start - pad_len
        if 0 <= orig_start < total_frames:
            frame = x_pad[next_start:next_start + n_fft, :]
            mono = np.sqrt(np.mean(frame**2, axis=1))
            level = rms_dbfs(mono)
            levels.append(level)
        next_start += hop
    
    return np.array(levels)


def simulate_gate_with_threshold(levels, threshold_dbfs, hyst_db, up_delay_frames=0):
    """用给定阈值模拟门控状态"""
    Ton = threshold_dbfs + hyst_db / 2
    Toff = threshold_dbfs - hyst_db / 2
    
    state = 1  # 1=C1, 2=C2
    pending_c2_at = None
    states = []
    
    for i, level in enumerate(levels):
        if state == 1:
            if level >= Ton:
                if pending_c2_at is None:
                    pending_c2_at = i + up_delay_frames
            else:
                pending_c2_at = None
            if pending_c2_at is not None and i >= pending_c2_at:
                state = 2
                pending_c2_at = None
        else:
            if level <= Toff:
                state = 1
                pending_c2_at = None
        
        states.append('C1' if state == 1 else 'C2')
    
    return states


def find_optimal_threshold(levels, hyst_db, target_c2_ratio=0.5, up_delay_frames=0):
    """
    二分查找最优门控阈值，使 C2_ratio 接近 target_c2_ratio
    
    返回: (optimal_threshold, achieved_c2_ratio)
    """
    # 初值：电平分布的中位数
    level_median = np.median(levels)
    level_min = np.min(levels)
    level_max = np.max(levels)
    
    # 二分搜索范围
    T_low = level_min - 10
    T_high = level_max + 10
    
    best_T = level_median
    best_ratio = 0.0
    best_diff = 1.0
    
    for iteration in range(30):  # 最多30次迭代
        T_mid = (T_low + T_high) / 2
        
        states = simulate_gate_with_threshold(levels, T_mid, hyst_db, up_delay_frames)
        c2_ratio = sum(1 for s in states if s == 'C2') / len(states)
        
        diff = abs(c2_ratio - target_c2_ratio)
        
        if diff < best_diff:
            best_diff = diff
            best_T = T_mid
            best_ratio = c2_ratio
        
        # 收敛条件
        if diff < 0.01:  # 误差 < 1%
            break
        
        # 二分调整
        if c2_ratio < target_c2_ratio:
            # C2太少，需要降低阈值
            T_high = T_mid
        else:
            # C2太多，需要升高阈值
            T_low = T_mid
    
    return best_T, best_ratio


def analyze_gate_stats(states, levels, sr, hop):
    """分析 Gate 统计"""
    n = len(states)
    if n == 0:
        return {}

    c2_count = sum(1 for s in states if s == 'C2')
    c2_ratio = c2_count / n

    switch_count = sum(1 for i in range(1, n) if states[i] != states[i-1])

    run_lengths = []
    current_run = 1
    for i in range(1, n):
        if states[i] == states[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    short_runs = sum(1 for r in run_lengths if r <= 3)
    short_run_ratio = short_runs / len(run_lengths) if run_lengths else 0

    duration_sec = n * hop / sr
    duration_min = duration_sec / 60
    switches_per_min = switch_count / duration_min if duration_min > 0 else 0

    c1_levels = [levels[i] for i in range(n) if states[i] == 'C1']
    c2_levels = [levels[i] for i in range(n) if states[i] == 'C2']

    return {
        'total_frames': n,
        'duration_min': duration_min,
        'c2_count': c2_count,
        'c2_ratio': c2_ratio,
        'switch_count': switch_count,
        'switches_per_min': switches_per_min,
        'run_min': min(run_lengths) if run_lengths else 0,
        'run_max': max(run_lengths) if run_lengths else 0,
        'run_median': np.median(run_lengths) if run_lengths else 0,
        'short_runs': short_runs,
        'short_run_ratio': short_run_ratio,
        'c1_level_mean': np.mean(c1_levels) if c1_levels else 0,
        'c2_level_mean': np.mean(c2_levels) if c2_levels else 0,
    }


# ============================================================
# C. 条件频谱验证（改进版：剔除弱能量帧 + 频带锚定）
# ============================================================

def find_stable_frames(states, margin=2):
    """找到稳定帧"""
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


def compute_conditional_spectrum_v2(x, y, sr, states, levels, n_fft, hop, 
                                     level_percentile=10, anchor_band=(900, 1100)):
    """
    改进版条件频谱计算
    
    改进点：
    1. 剔除弱能量帧（低于 level_percentile 分位数）
    2. 使用频带锚定（anchor_band）而非单点
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    ch = x.shape[1]
    pad_len = n_fft // 2

    x_pad = np.vstack([np.zeros((pad_len, ch), dtype=x.dtype), x, np.zeros((pad_len, ch), dtype=x.dtype)])
    y_pad = np.vstack([np.zeros((pad_len, ch), dtype=y.dtype), y, np.zeros((pad_len, ch), dtype=y.dtype)])

    # 计算电平阈值（剔除最低 level_percentile% 的帧）
    level_threshold = np.percentile(levels, level_percentile)

    c1_stable, c2_stable = find_stable_frames(states, margin=2)

    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    n_bins = len(freqs)
    win = np.hanning(n_fft).astype(np.float32)
    
    # 锚定频带的 bin 索引
    anchor_mask = (freqs >= anchor_band[0]) & (freqs <= anchor_band[1])

    c1_ratios = []
    c2_ratios = []
    c1_used = 0
    c2_used = 0

    for idx in c1_stable:
        if levels[idx] < level_threshold:
            continue
        orig_start = idx * hop
        if orig_start < 0 or orig_start + n_fft > len(x):
            continue
        start = orig_start + pad_len
        frame_x = x_pad[start:start + n_fft, :]
        frame_y = y_pad[start:start + n_fft, :]
        X = np.zeros(n_bins, dtype=np.float32)
        Y = np.zeros(n_bins, dtype=np.float32)
        for c in range(ch):
            X += np.abs(np.fft.rfft(frame_x[:, c] * win))
            Y += np.abs(np.fft.rfft(frame_y[:, c] * win))
        X /= ch
        Y /= ch
        X = np.maximum(X, 1e-10)
        ratio = Y / X
        # 频带锚定：归一化到 anchor_band 的增益为 1
        anchor_gain = np.mean(ratio[anchor_mask])
        if anchor_gain > 0:
            ratio = ratio / anchor_gain
        c1_ratios.append(ratio)
        c1_used += 1

    for idx in c2_stable:
        if levels[idx] < level_threshold:
            continue
        orig_start = idx * hop
        if orig_start < 0 or orig_start + n_fft > len(x):
            continue
        start = orig_start + pad_len
        frame_x = x_pad[start:start + n_fft, :]
        frame_y = y_pad[start:start + n_fft, :]
        X = np.zeros(n_bins, dtype=np.float32)
        Y = np.zeros(n_bins, dtype=np.float32)
        for c in range(ch):
            X += np.abs(np.fft.rfft(frame_x[:, c] * win))
            Y += np.abs(np.fft.rfft(frame_y[:, c] * win))
        X /= ch
        Y /= ch
        X = np.maximum(X, 1e-10)
        ratio = Y / X
        anchor_gain = np.mean(ratio[anchor_mask])
        if anchor_gain > 0:
            ratio = ratio / anchor_gain
        c2_ratios.append(ratio)
        c2_used += 1

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

    return freqs, c1_db, c2_db, c1_used, c2_used


def compute_spectrum_metrics_v2(freqs, c1_db, c2_db, c1_theory, c2_theory, fc, gain_limit):
    """计算频谱验收指标（更严格的纯数字音乐标准）"""
    metrics = {}
    
    # 平台段 RMSE (低频: <420Hz, 高频: >2380Hz for ±15dB)
    f_lo_platform = fc * 2 ** (-gain_limit / 12)  # ~420 Hz
    f_hi_platform = fc * 2 ** (gain_limit / 12)   # ~2380 Hz
    
    # 低频平台段
    lo_mask = (freqs >= 100) & (freqs <= f_lo_platform * 0.8)
    if np.any(lo_mask):
        c1_lo_rmse = np.sqrt(np.mean((c1_db[lo_mask] - c1_theory[lo_mask])**2))
        c2_lo_rmse = np.sqrt(np.mean((c2_db[lo_mask] - c2_theory[lo_mask])**2))
        metrics['c1_lo_platform_rmse'] = c1_lo_rmse
        metrics['c2_lo_platform_rmse'] = c2_lo_rmse
        metrics['c1_lo_platform_mean'] = np.mean(c1_db[lo_mask])
        metrics['c2_lo_platform_mean'] = np.mean(c2_db[lo_mask])
    
    # 高频平台段
    hi_mask = (freqs >= f_hi_platform * 1.2) & (freqs <= 10000)
    if np.any(hi_mask):
        c1_hi_rmse = np.sqrt(np.mean((c1_db[hi_mask] - c1_theory[hi_mask])**2))
        c2_hi_rmse = np.sqrt(np.mean((c2_db[hi_mask] - c2_theory[hi_mask])**2))
        metrics['c1_hi_platform_rmse'] = c1_hi_rmse
        metrics['c2_hi_platform_rmse'] = c2_hi_rmse
        metrics['c1_hi_platform_mean'] = np.mean(c1_db[hi_mask])
        metrics['c2_hi_platform_mean'] = np.mean(c2_db[hi_mask])
    
    # 斜坡段 RMSE
    slope_mask = (freqs >= f_lo_platform * 1.2) & (freqs <= f_hi_platform * 0.8)
    if np.any(slope_mask):
        c1_slope_rmse = np.sqrt(np.mean((c1_db[slope_mask] - c1_theory[slope_mask])**2))
        c2_slope_rmse = np.sqrt(np.mean((c2_db[slope_mask] - c2_theory[slope_mask])**2))
        metrics['c1_slope_rmse'] = c1_slope_rmse
        metrics['c2_slope_rmse'] = c2_slope_rmse
    
    # fc 附近误差（900-1100Hz 频带平均）
    fc_mask = (freqs >= 900) & (freqs <= 1100)
    if np.any(fc_mask):
        metrics['c1_fc_error'] = abs(np.mean(c1_db[fc_mask]))
        metrics['c2_fc_error'] = abs(np.mean(c2_db[fc_mask]))
    
    return metrics


# ============================================================
# D. 效果量化 - Tilt Index
# ============================================================

def compute_tilt_index(x, y, sr, states, levels, n_fft, hop, level_percentile=10):
    """计算 Tilt Index，剔除弱能量帧"""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    ch = x.shape[1]
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    win = np.hanning(n_fft).astype(np.float32)
    
    lo_mask = (freqs >= 200) & (freqs <= 1000)
    hi_mask = (freqs >= 2000) & (freqs <= 8000)
    
    level_threshold = np.percentile(levels, level_percentile)
    
    ti_input = []
    ti_output = []
    ti_c1 = []
    ti_c2 = []
    
    n_frames = len(states)
    
    for i in range(n_frames):
        if levels[i] < level_threshold:
            continue
        orig_start = i * hop
        if orig_start + n_fft > len(x):
            break
        
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
            results[f'{key}_n'] = len(arr)
    
    if 'c1_mean' in results and 'c2_mean' in results:
        results['ti_effect'] = results['c2_mean'] - results['c1_mean']
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Tomatis ±15dB 自适应验证工具 v2')
    parser.add_argument('-i', '--input', required=True, help='原始输入音频')
    parser.add_argument('-o', '--output', required=True, help='处理后输出音频')

    # 纯数字音乐默认参数
    parser.add_argument('--hyst_db', type=float, default=1.0, help='回差 dB')
    parser.add_argument('--up_delay_ms', type=float, default=0, help='上行延迟 ms')
    parser.add_argument('--target_c2', type=float, default=0.5, help='目标 C2 占比')

    # 滤波器参数
    parser.add_argument('--fc', type=float, default=1000)
    parser.add_argument('--slope', type=float, default=12)
    parser.add_argument('--c1_low', type=float, default=15.0)
    parser.add_argument('--c1_high', type=float, default=-15.0)
    parser.add_argument('--c2_low', type=float, default=-15.0)
    parser.add_argument('--c2_high', type=float, default=15.0)

    # STFT 参数
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--hop', type=int, default=2048)
    
    # 验证参数
    parser.add_argument('--level_percentile', type=float, default=10, help='剔除最低 N% 能量帧')

    # 输出
    parser.add_argument('--out_prefix', default='verify_15db_v2')

    args = parser.parse_args()

    print('=' * 70)
    print('Tomatis ±15dB 自适应验证工具 v2')
    print('=' * 70)
    print()
    
    gain_limit = abs(args.c1_low)
    
    print(f'滤波器参数:')
    print(f'  fc={args.fc} Hz, slope={args.slope} dB/oct')
    print(f'  C1: +{args.c1_low}/{args.c1_high} dB, C2: {args.c2_low}/+{args.c2_high} dB')
    print(f'  低频封顶: ~{args.fc * 2**(-gain_limit/args.slope):.0f} Hz')
    print(f'  高频封顶: ~{args.fc * 2**(gain_limit/args.slope):.0f} Hz')
    print()
    print(f'门控参数:')
    print(f'  hyst={args.hyst_db} dB, delay={args.up_delay_ms} ms')
    print(f'  目标 C2 占比: {args.target_c2*100:.0f}%')
    print()

    all_pass = True
    report_lines = []
    report_lines.append('Tomatis ±15dB 自适应验证报告 v2')
    report_lines.append('=' * 50)

    # ========================================
    # A. 工程检查
    # ========================================
    print('-' * 50)
    print('A. 工程检查')
    print('-' * 50)

    eng = check_engineering(args.input, args.output)
    eng_pass = eng['sr_match'] and eng['ch_match'] and eng['frames_match'] and eng['peak_safe'] and eng['dc_safe']

    print(f'  采样率: {eng["sr_in"]} -> {eng["sr_out"]} {"PASS" if eng["sr_match"] else "FAIL"}')
    print(f'  声道数: {eng["ch_in"]} -> {eng["ch_out"]} {"PASS" if eng["ch_match"] else "FAIL"}')
    print(f'  样点数: {eng["frames_in"]} -> {eng["frames_out"]} (diff={eng["frames_diff"]}) {"PASS" if eng["frames_match"] else "FAIL"}')
    print(f'  峰值: {eng["peak"]:.4f} ({eng["peak_dbfs"]:.2f} dBFS) {"PASS" if eng["peak_safe"] else "FAIL"}')
    print(f'  DC偏移: {eng["dc_mean"]:.6f} {"PASS" if eng["dc_safe"] else "FAIL"}')
    print(f'  结果: {"PASS" if eng_pass else "FAIL"}')

    if not eng_pass:
        all_pass = False

    report_lines.append('\nA. 工程检查')
    report_lines.append(f'  结果: {"PASS" if eng_pass else "FAIL"}')
    report_lines.append(f'  峰值: {eng["peak"]:.4f}')

    # ========================================
    # B. 自适应门控阈值求解
    # ========================================
    print()
    print('-' * 50)
    print('B. 自适应门控阈值求解')
    print('-' * 50)

    print('  读取输入音频...')
    x, sr = sf.read(args.input, dtype='float32')
    y, _ = sf.read(args.output, dtype='float32')
    
    print('  计算帧电平...')
    levels = compute_frame_levels(x, sr, args.n_fft, args.hop)
    
    level_median = np.median(levels)
    level_p10 = np.percentile(levels, 10)
    level_p90 = np.percentile(levels, 90)
    
    print(f'  电平分布: p10={level_p10:.1f}, median={level_median:.1f}, p90={level_p90:.1f} dBFS')
    
    up_delay_frames = int(args.up_delay_ms * sr / 1000 / args.hop)
    
    print('  二分查找最优阈值...')
    optimal_T, achieved_c2 = find_optimal_threshold(
        levels, args.hyst_db, args.target_c2, up_delay_frames
    )
    
    print(f'  最优阈值 T: {optimal_T:.2f} dBFS')
    print(f'  达成 C2 占比: {achieved_c2*100:.1f}%')
    
    # 用最优阈值重新模拟状态
    states = simulate_gate_with_threshold(levels, optimal_T, args.hyst_db, up_delay_frames)
    
    stats = analyze_gate_stats(states, levels, sr, args.hop)
    
    c2_ratio_ok = 0.48 <= stats['c2_ratio'] <= 0.52
    print(f'  C2 占比验证 (48%-52%): {"PASS" if c2_ratio_ok else "FAIL"}')
    print(f'  切换次数: {stats["switch_count"]} ({stats["switches_per_min"]:.1f}/min)')
    print(f'  C1/C2 平均电平: {stats["c1_level_mean"]:.1f} / {stats["c2_level_mean"]:.1f} dBFS')

    report_lines.append('\nB. 自适应门控')
    report_lines.append(f'  最优阈值 T: {optimal_T:.2f} dBFS')
    report_lines.append(f'  C2 占比: {achieved_c2*100:.1f}%')
    report_lines.append(f'  切换次数: {stats["switch_count"]}')

    # ========================================
    # C. 条件频谱验证
    # ========================================
    print()
    print('-' * 50)
    print('C. 条件频谱验证 (改进版)')
    print('-' * 50)
    
    print(f'  剔除最低 {args.level_percentile}% 能量帧')
    print(f'  使用 900-1100Hz 频带锚定')

    freqs, c1_db, c2_db, c1_n, c2_n = compute_conditional_spectrum_v2(
        x, y, sr, states, levels, args.n_fft, args.hop,
        level_percentile=args.level_percentile, anchor_band=(900, 1100)
    )

    print(f'  有效帧: C1={c1_n}, C2={c2_n}')

    # 理论曲线
    c1_theory = build_tilt_gain_db(freqs, args.fc, args.slope, args.c1_low, args.c1_high)
    c2_theory = build_tilt_gain_db(freqs, args.fc, args.slope, args.c2_low, args.c2_high)

    metrics = compute_spectrum_metrics_v2(freqs, c1_db, c2_db, c1_theory, c2_theory, args.fc, gain_limit)

    # 输出指标
    print(f'  低频平台 (100-350Hz):')
    print(f'    C1: {metrics.get("c1_lo_platform_mean", 0):.1f} dB (目标 +{gain_limit}), RMSE={metrics.get("c1_lo_platform_rmse", 0):.2f}')
    print(f'    C2: {metrics.get("c2_lo_platform_mean", 0):.1f} dB (目标 -{gain_limit}), RMSE={metrics.get("c2_lo_platform_rmse", 0):.2f}')
    
    print(f'  高频平台 (3000-10000Hz):')
    print(f'    C1: {metrics.get("c1_hi_platform_mean", 0):.1f} dB (目标 -{gain_limit}), RMSE={metrics.get("c1_hi_platform_rmse", 0):.2f}')
    print(f'    C2: {metrics.get("c2_hi_platform_mean", 0):.1f} dB (目标 +{gain_limit}), RMSE={metrics.get("c2_hi_platform_rmse", 0):.2f}')
    
    print(f'  斜坡段 RMSE:')
    print(f'    C1: {metrics.get("c1_slope_rmse", 0):.2f} dB, C2: {metrics.get("c2_slope_rmse", 0):.2f} dB')
    
    print(f'  fc (1000Hz) 误差:')
    print(f'    C1: {metrics.get("c1_fc_error", 0):.2f} dB, C2: {metrics.get("c2_fc_error", 0):.2f} dB')

    # 验收判定（纯数字音乐更严格）
    platform_rmse_ok = (metrics.get('c1_lo_platform_rmse', 99) < 0.5 and 
                        metrics.get('c2_lo_platform_rmse', 99) < 0.5 and
                        metrics.get('c1_hi_platform_rmse', 99) < 0.5 and 
                        metrics.get('c2_hi_platform_rmse', 99) < 0.5)
    slope_rmse_ok = (metrics.get('c1_slope_rmse', 99) < 1.0 and 
                     metrics.get('c2_slope_rmse', 99) < 1.0)
    fc_ok = (metrics.get('c1_fc_error', 99) < 0.5 and 
             metrics.get('c2_fc_error', 99) < 0.5)
    
    spectrum_pass = platform_rmse_ok and slope_rmse_ok and fc_ok
    
    print(f'  平台 RMSE (<0.5dB): {"PASS" if platform_rmse_ok else "FAIL"}')
    print(f'  斜坡 RMSE (<1.0dB): {"PASS" if slope_rmse_ok else "FAIL"}')
    print(f'  fc 误差 (<0.5dB): {"PASS" if fc_ok else "FAIL"}')
    print(f'  条件频谱结果: {"PASS" if spectrum_pass else "FAIL"}')

    if not spectrum_pass:
        all_pass = False

    report_lines.append('\nC. 条件频谱验证')
    report_lines.append(f'  有效帧: C1={c1_n}, C2={c2_n}')
    report_lines.append(f'  平台 RMSE: {"PASS" if platform_rmse_ok else "FAIL"}')
    report_lines.append(f'  斜坡 RMSE: {"PASS" if slope_rmse_ok else "FAIL"}')
    report_lines.append(f'  fc 误差: {"PASS" if fc_ok else "FAIL"}')

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
    # D. 效果量化
    # ========================================
    print()
    print('-' * 50)
    print('D. 效果量化 (Tilt Index)')
    print('-' * 50)

    ti_data = compute_tilt_index(x, y, sr, states, levels, args.n_fft, args.hop, 
                                  level_percentile=args.level_percentile)
    ti_results = analyze_tilt_index(ti_data)

    print(f'  输入 TI (n={ti_results.get("input_n", 0)}): mean={ti_results.get("input_mean", 0):.2f}')
    print(f'  输出 TI (n={ti_results.get("output_n", 0)}): mean={ti_results.get("output_mean", 0):.2f}')
    print(f'  C1 TI (n={ti_results.get("c1_n", 0)}): mean={ti_results.get("c1_mean", 0):.2f}')
    print(f'  C2 TI (n={ti_results.get("c2_n", 0)}): mean={ti_results.get("c2_mean", 0):.2f}')
    
    ti_effect = ti_results.get('ti_effect', 0)
    print(f'  分离度 (C2-C1): {ti_effect:.2f} dB')
    
    # ±15dB 理论分离度应该很大
    ti_effect_ok = ti_effect > 20.0  # 期望至少 20 dB
    print(f'  分离度验证 (>20dB): {"PASS" if ti_effect_ok else f"WARN ({ti_effect:.1f}dB)"}')

    report_lines.append('\nD. 效果量化')
    report_lines.append(f'  C1 TI: {ti_results.get("c1_mean", 0):.2f} dB')
    report_lines.append(f'  C2 TI: {ti_results.get("c2_mean", 0):.2f} dB')
    report_lines.append(f'  分离度: {ti_effect:.2f} dB')

    # ========================================
    # 绘图
    # ========================================
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # C1
        ax = axes[0]
        ax.semilogx(freqs, c1_db, 'b-', label='C1 measured', alpha=0.7, linewidth=1.5)
        ax.semilogx(freqs, c1_theory, 'b--', label='C1 theory', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(gain_limit, color='green', linestyle=':', alpha=0.5)
        ax.axhline(-gain_limit, color='red', linestyle=':', alpha=0.5)
        ax.axvline(args.fc, color='orange', linestyle='--', alpha=0.7, label=f'fc={args.fc}Hz')
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 20)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'C1 条件频谱 (n={c1_n}, 剔除<p{args.level_percentile}能量帧)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # C2
        ax = axes[1]
        ax.semilogx(freqs, c2_db, 'r-', label='C2 measured', alpha=0.7, linewidth=1.5)
        ax.semilogx(freqs, c2_theory, 'r--', label='C2 theory', linewidth=2)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(gain_limit, color='green', linestyle=':', alpha=0.5)
        ax.axhline(-gain_limit, color='red', linestyle=':', alpha=0.5)
        ax.axvline(args.fc, color='orange', linestyle='--', alpha=0.7, label=f'fc={args.fc}Hz')
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 20)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Gain (dB)')
        ax.set_title(f'C2 条件频谱 (n={c2_n}, 剔除<p{args.level_percentile}能量帧)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        spectrum_png = f'{args.out_prefix}_spectrum.png'
        plt.savefig(spectrum_png, dpi=150)
        print(f'  频谱图已保存: {spectrum_png}')
        plt.close()

        # Tilt Index 图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(ti_data['input'], bins=50, alpha=0.5, label='Input', color='gray')
        ax.hist(ti_data['output'], bins=50, alpha=0.5, label='Output', color='blue')
        ax.set_xlabel('Tilt Index (dB)')
        ax.set_ylabel('Count')
        ax.set_title('Tilt Index: Input vs Output')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if len(ti_data['c1']) > 0:
            ax.hist(ti_data['c1'], bins=50, alpha=0.5, label=f'C1 (mean={ti_results.get("c1_mean", 0):.1f})', color='blue')
        if len(ti_data['c2']) > 0:
            ax.hist(ti_data['c2'], bins=50, alpha=0.5, label=f'C2 (mean={ti_results.get("c2_mean", 0):.1f})', color='red')
        ax.set_xlabel('Tilt Index (dB)')
        ax.set_ylabel('Count')
        ax.set_title(f'Tilt Index: C1 vs C2 (分离度={ti_effect:.1f} dB)')
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

    print(f'  A. 工程检查: {"PASS" if eng_pass else "FAIL"}')
    print(f'  B. 门控 C2 占比 ({achieved_c2*100:.0f}%): {"PASS" if c2_ratio_ok else "FAIL"}')
    print(f'  C. 条件频谱: {"PASS" if spectrum_pass else "FAIL"}')
    print(f'  D. TI 分离度 ({ti_effect:.0f}dB): {"PASS" if ti_effect_ok else "WARN"}')

    print()
    if all_pass and c2_ratio_ok:
        print('验证结果: PASS')
        print('D_MNF_matched_15dB.flac 符合 Tomatis ±15dB 规则')
    else:
        print('验证结果: FAIL')
        print('请检查上述 FAIL 项')

    report_lines.append('\n' + '=' * 50)
    report_lines.append(f'总体结果: {"PASS" if all_pass and c2_ratio_ok else "FAIL"}')
    
    report_path = f'{args.out_prefix}_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\n报告已保存: {report_path}')

    return 0 if (all_pass and c2_ratio_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
