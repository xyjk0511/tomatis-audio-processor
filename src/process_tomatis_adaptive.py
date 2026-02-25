#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tomatis 自适应处理器 v2

核心特性：
1. 预衰减 headroom：atten_db = max(0, peak_dbfs + 15 + margin)
2. 自适应阈值 T 求解：目标 C2=50%
3. min_hold 最短保持时间：防止切换过密
4. 输出 state_csv 供验证

推荐参数（纯数字音乐）：
- hyst_db = 3.0 dB
- min_hold_ms = 250 ms（约6帧@hop=2048）

作者: DSP 分析工具
日期: 2026-01-20
"""

import argparse
import csv
import sys
import numpy as np
import soundfile as sf

import math

EPS = 1e-12
PEAK_LIMIT = 0.999


def rms_dbfs(x_mono: np.ndarray) -> float:
    r = np.sqrt(np.mean(x_mono * x_mono) + EPS)
    return float(20.0 * np.log10(r + EPS))


def db_to_lin(db):
    return 10 ** (np.asarray(db) / 20.0)


def build_tilt_gain_db(freqs, fc, slope_db_per_oct, low_gain_db, high_gain_db):
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


def compute_frame_levels(x, sr, n_fft, hop, silence_threshold=-70):
    """计算每帧电平，返回 (levels, valid_mask)"""
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

    levels = np.array(levels)
    valid_mask = levels > silence_threshold
    # 时间从hop/sr开始，每帧递增hop/sr
    frame_sec = hop / sr
    times = [(i + 1) * frame_sec for i in range(len(levels))]
    return levels, valid_mask, times


def simulate_gate(levels, threshold_dbfs, hyst_db=3.0, min_hold_frames=6):
    """
    模拟门控状态机（带 min_hold 最短保持）
    
    参数：
        levels: 每帧电平 dBFS
        threshold_dbfs: 门控阈值
        hyst_db: 回差（默认 3.0 dB）
        min_hold_frames: 切换后最短保持帧数（默认 6 帧 ≈ 250ms）
    """
    Ton = threshold_dbfs + hyst_db / 2
    Toff = threshold_dbfs - hyst_db / 2
    
    state = 1  # 1=C1, 2=C2
    states = []
    frames_since_switch = min_hold_frames  # 初始允许切换
    
    for level in levels:
        # 更新距上次切换的帧数
        frames_since_switch += 1
        
        # 只有超过 min_hold 才允许切换
        if frames_since_switch >= min_hold_frames:
            if state == 1:
                if level >= Ton:
                    state = 2
                    frames_since_switch = 0  # 重置计数
            else:
                if level <= Toff:
                    state = 1
                    frames_since_switch = 0  # 重置计数
        
        states.append('C1' if state == 1 else 'C2')
    
    return states


def find_optimal_threshold(levels, valid_mask, hyst_db=3.0, min_hold_frames=6, target_c2=0.5):
    """二分查找最优阈值使 C2=target_c2（使用带 min_hold 的状态机）"""
    valid_levels = levels[valid_mask]
    if len(valid_levels) == 0:
        return np.median(levels)
    
    T_low = np.percentile(valid_levels, 5)
    T_high = np.percentile(valid_levels, 95)
    
    best_T = np.median(valid_levels)
    best_diff = 1.0
    
    for _ in range(30):
        T_mid = (T_low + T_high) / 2
        states = simulate_gate(levels, T_mid, hyst_db, min_hold_frames)
        c2_ratio = sum(1 for s in states if s == 'C2') / len(states)
        
        diff = abs(c2_ratio - target_c2)
        if diff < best_diff:
            best_diff = diff
            best_T = T_mid
        
        if diff < 0.01:
            break
        
        if c2_ratio < target_c2:
            T_high = T_mid
        else:
            T_low = T_mid
    
    return best_T


def process(
    in_path,
    out_path,
    fc=1000.0,
    slope=12.0,
    c1_low=15.0, c1_high=-15.0,
    c2_low=-15.0, c2_high=15.0,
    target_c2=0.5,
    hyst_db=3.0,
    min_hold_ms=250.0,
    xfade_ms=500.0,
    headroom_margin=2.0,
    n_fft=4096,
    hop=2048,
    state_csv_path=None
):
    print("=" * 60)
    print("Tomatis 自适应处理器")
    print("=" * 60)
    
    # 1. 读取输入
    print(f"\n读取输入: {in_path}")
    x, sr = sf.read(in_path, dtype='float32')
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ch = x.shape[1]
    total_frames = len(x)
    
    print(f"  采样率: {sr} Hz")
    print(f"  声道数: {ch}")
    print(f"  时长: {total_frames / sr:.2f} s")
    
    # 2. 计算 min_hold_frames 和 xfade_frames
    frame_ms = hop / sr * 1000
    min_hold_frames = int(np.ceil(min_hold_ms / frame_ms))
    xfade_frames = int(np.ceil(xfade_ms / frame_ms))
    
    print(f"\n门控参数:")
    print(f"  hyst_db: {hyst_db} dB")
    print(f"  min_hold: {min_hold_ms} ms ({min_hold_frames} 帧)")
    print(f"  xfade: {xfade_ms} ms ({xfade_frames} 帧)")
    print(f"  帧时长: {frame_ms:.2f} ms")
    
    # 3. 计算预衰减
    input_peak = np.max(np.abs(x))
    input_peak_dbfs = 20 * np.log10(input_peak + EPS)
    max_gain = max(abs(c1_low), abs(c2_high))
    
    atten_db = max(0, input_peak_dbfs + max_gain + headroom_margin)
    atten_lin = db_to_lin(-atten_db)
    
    print(f"\n预衰减计算:")
    print(f"  输入峰值: {input_peak_dbfs:.2f} dBFS")
    print(f"  最大增益: +{max_gain} dB")
    print(f"  余量: {headroom_margin} dB")
    print(f"  预衰减: {-atten_db:.2f} dB")
    
    # 3. 应用预衰减
    x_atten = x * atten_lin
    
    # 5. 计算帧电平 + 自适应阈值
    print(f"\n自适应门控:")
    levels, valid_mask, times = compute_frame_levels(x_atten, sr, n_fft, hop)
    
    valid_count = np.sum(valid_mask)
    print(f"  总帧数: {len(levels)}")
    print(f"  有效帧: {valid_count} ({valid_count/len(levels)*100:.1f}%)")
    
    optimal_T = find_optimal_threshold(levels, valid_mask, hyst_db, min_hold_frames, target_c2)
    states = simulate_gate(levels, optimal_T, hyst_db, min_hold_frames)
    
    c2_ratio = sum(1 for s in states if s == 'C2') / len(states)
    switch_count = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
    duration_min = total_frames / sr / 60
    switches_per_min = switch_count / duration_min if duration_min > 0 else 0
    
    # 计算短段比例
    run_lengths = []
    current_run = 1
    for i in range(1, len(states)):
        if states[i] == states[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    short_runs = sum(1 for r in run_lengths if r < min_hold_frames)
    short_run_ratio = short_runs / len(run_lengths) if run_lengths else 0
    
    print(f"  最优阈值 T: {optimal_T:.2f} dBFS")
    print(f"  C2 占比: {c2_ratio*100:.1f}%")
    print(f"  切换次数: {switch_count} ({switches_per_min:.1f}/min)")
    print(f"  短段比例: {short_run_ratio*100:.1f}%")
    
    # 6. 计算 alpha(t) 平滑控制量
    # 目标 alpha: C1=0, C2=1
    target_alpha = np.array([0.0 if s == 'C1' else 1.0 for s in states])
    
    # 平滑 alpha：每帧最多变化 1/xfade_frames
    alpha = np.zeros_like(target_alpha)
    alpha[0] = target_alpha[0]
    step = 1.0 / xfade_frames if xfade_frames > 0 else 1.0
    
    for i in range(1, len(alpha)):
        diff = target_alpha[i] - alpha[i-1]
        if abs(diff) <= step:
            alpha[i] = target_alpha[i]
        else:
            alpha[i] = alpha[i-1] + step * np.sign(diff)
    
    print(f"\nCrossfade 统计:")
    print(f"  alpha 范围: {alpha.min():.3f} - {alpha.max():.3f}")
    print(f"  平均 alpha: {alpha.mean():.3f}")
    
    # 7. 构建滤波器
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    c1_gain_db = build_tilt_gain_db(freqs, fc, slope, c1_low, c1_high)
    c2_gain_db = build_tilt_gain_db(freqs, fc, slope, c2_low, c2_high)
    
    print(f"\n滤波器参数:")
    print(f"  fc: {fc} Hz")
    print(f"  slope: {slope} dB/oct")
    print(f"  C1: +{c1_low}/{c1_high} dB")
    print(f"  C2: {c2_low}/+{c2_high} dB")
    
    # 8. STFT 处理（带 crossfade）
    print(f"\n处理中...")
    win = np.hanning(n_fft).astype(np.float32)
    
    pad_len = n_fft // 2
    x_pad = np.vstack([np.zeros((pad_len, ch), dtype=x_atten.dtype), 
                       x_atten, 
                       np.zeros((pad_len, ch), dtype=x_atten.dtype)])
    
    y = np.zeros_like(x_atten)
    norm = np.zeros(total_frames, dtype=np.float32)
    
    next_start = 0
    frame_idx = 0
    
    while next_start + n_fft <= len(x_pad):
        orig_start = next_start - pad_len
        if 0 <= orig_start < total_frames and frame_idx < len(states):
            # 在 dB 域混合增益曲线
            a = alpha[frame_idx]
            mixed_gain_db = (1 - a) * c1_gain_db + a * c2_gain_db
            gain = db_to_lin(mixed_gain_db).astype(np.float32)
            
            # 处理每个声道
            frame = x_pad[next_start:next_start + n_fft, :]
            y_frame = np.zeros_like(frame)
            
            for c in range(ch):
                X = np.fft.rfft(frame[:, c] * win)
                X *= gain
                y_frame[:, c] = np.fft.irfft(X, n_fft) * win
            
            # Overlap-add
            write_start = max(0, orig_start)
            write_end = min(total_frames, orig_start + n_fft)
            
            frame_start = write_start - orig_start
            frame_end = write_end - orig_start
            
            y[write_start:write_end] += y_frame[frame_start:frame_end]
            norm[write_start:write_end] += win[frame_start:frame_end] ** 2
            
            frame_idx += 1
        
        next_start += hop
    
    # 归一化
    norm = np.maximum(norm, 1e-8)
    for c in range(ch):
        y[:, c] /= norm
    
    # 7. 恢复预衰减（关键！否则输出电平会低 atten_db）
    if atten_db > 0:
        restore_lin = db_to_lin(atten_db)
        y *= restore_lin
        print(f"  恢复预衰减: +{atten_db:.2f} dB")
    
    # 8. 峰值保护
    output_peak = np.max(np.abs(y))
    if output_peak > PEAK_LIMIT:
        scale = PEAK_LIMIT / output_peak
        y *= scale
        print(f"  峰值保护: 缩放 {20*np.log10(scale):.2f} dB")
    
    output_peak_final = np.max(np.abs(y))
    print(f"  输出峰值: {20*np.log10(output_peak_final + EPS):.2f} dBFS")
    
    # 8. 保存输出
    sf.write(out_path, y, sr, subtype='PCM_24')
    print(f"\n输出已保存: {out_path}")
    
    # 10. 保存 state_csv（带 alpha 列）
    if state_csv_path:
        with open(state_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'time_sec', 'level_dbfs', 'state', 'alpha'])
            for i, (t, lvl, st) in enumerate(zip(times, levels, states)):
                a = alpha[i] if i < len(alpha) else 0
                writer.writerow([i+1, f'{t:.6f}', f'{lvl:.4f}', st, f'{a:.4f}'])
        print(f"状态已保存: {state_csv_path}")
    
    # 10. 统计信息
    print(f"\n处理统计:")
    print(f"  预衰减: {-atten_db:.2f} dB")
    print(f"  最优阈值 T: {optimal_T:.2f} dBFS")
    print(f"  C2 占比: {c2_ratio*100:.1f}%")
    print(f"  切换次数: {switch_count} ({switches_per_min:.1f}/min)")
    print(f"  短段比例: {short_run_ratio*100:.1f}%")
    print(f"  输出峰值: {20*np.log10(output_peak_final + EPS):.2f} dBFS")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Tomatis 自适应处理器')
    parser.add_argument('-i', '--input', required=True, help='输入音频')
    parser.add_argument('-o', '--output', required=True, help='输出音频')
    parser.add_argument('--state_csv', help='状态 CSV 输出路径')
    
    # 滤波器参数
    parser.add_argument('--fc', type=float, default=1000)
    parser.add_argument('--slope', type=float, default=12)
    parser.add_argument('--c1_low', type=float, default=15.0)
    parser.add_argument('--c1_high', type=float, default=-15.0)
    parser.add_argument('--c2_low', type=float, default=-15.0)
    parser.add_argument('--c2_high', type=float, default=15.0)
    
    # 门控参数
    parser.add_argument('--target_c2', type=float, default=0.5, help='目标 C2 占比')
    parser.add_argument('--hyst_db', type=float, default=3.0, help='回差 dB（默认 3.0）')
    parser.add_argument('--min_hold_ms', type=float, default=250.0, help='最短保持 ms（默认 250）')
    parser.add_argument('--xfade_ms', type=float, default=500.0, help='Crossfade 过渡时间 ms（默认 500）')
    parser.add_argument('--headroom_margin', type=float, default=2.0, help='预衰减余量 dB')
    
    # STFT 参数
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--hop', type=int, default=2048)
    
    args = parser.parse_args()
    
    return process(
        args.input,
        args.output,
        fc=args.fc,
        slope=args.slope,
        c1_low=args.c1_low,
        c1_high=args.c1_high,
        c2_low=args.c2_low,
        c2_high=args.c2_high,
        target_c2=args.target_c2,
        hyst_db=args.hyst_db,
        min_hold_ms=args.min_hold_ms,
        xfade_ms=args.xfade_ms,
        headroom_margin=args.headroom_margin,
        n_fft=args.n_fft,
        hop=args.hop,
        state_csv_path=args.state_csv
    )


if __name__ == '__main__':
    sys.exit(main())
