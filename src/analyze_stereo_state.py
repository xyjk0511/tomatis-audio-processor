#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
立体声状态分析器
分别计算左右声道的dBFS和C1/C2状态
"""

import argparse
import csv
import numpy as np
import soundfile as sf

EPS = 1e-12


def rms_dbfs(x_mono):
    """计算单声道RMS dBFS"""
    r = np.sqrt(np.mean(x_mono * x_mono) + EPS)
    return float(20.0 * np.log10(r + EPS))


def format_time(seconds):
    """格式化为 分:秒"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def simulate_gate(levels, threshold_dbfs, hyst_db=3.0, min_hold_frames=6):
    """模拟门控状态机"""
    Ton = threshold_dbfs + hyst_db / 2
    Toff = threshold_dbfs - hyst_db / 2

    state = 1  # 1=C1, 2=C2
    states = []
    frames_since_switch = min_hold_frames

    for level in levels:
        frames_since_switch += 1
        if frames_since_switch >= min_hold_frames:
            if state == 1 and level >= Ton:
                state = 2
                frames_since_switch = 0
            elif state == 2 and level <= Toff:
                state = 1
                frames_since_switch = 0
        states.append('C1' if state == 1 else 'C2')

    return states


def find_optimal_threshold(levels, target_c2=0.5, hyst_db=3.0, min_hold_frames=6):
    """二分查找最优阈值"""
    valid_levels = levels[levels > -70]
    if len(valid_levels) == 0:
        return np.median(levels)

    T_low = np.percentile(valid_levels, 5)
    T_high = np.percentile(valid_levels, 95)
    best_T = np.median(valid_levels)

    for _ in range(30):
        T_mid = (T_low + T_high) / 2
        states = simulate_gate(levels, T_mid, hyst_db, min_hold_frames)
        c2_ratio = sum(1 for s in states if s == 'C2') / len(states)

        if abs(c2_ratio - target_c2) < 0.01:
            return T_mid

        if c2_ratio < target_c2:
            T_high = T_mid
        else:
            T_low = T_mid
        best_T = T_mid

    return best_T


def analyze(in_path, out_csv, target_c2=0.5, hyst_db=3.0, min_hold_ms=250.0, n_fft=4096, hop=2048):
    print(f"读取: {in_path}")
    x, sr = sf.read(in_path, dtype='float32')

    if x.ndim == 1:
        print("错误: 输入是单声道，需要立体声文件")
        return 1

    ch = x.shape[1]
    print(f"采样率: {sr} Hz, 声道: {ch}")

    # 计算帧参数
    frame_ms = hop / sr * 1000
    min_hold_frames = int(np.ceil(min_hold_ms / frame_ms))

    # Padding
    pad_len = n_fft // 2
    x_pad = np.vstack([np.zeros((pad_len, ch), dtype=x.dtype), x, np.zeros((pad_len, ch), dtype=x.dtype)])

    # 计算每帧每声道的dBFS
    left_levels = []
    right_levels = []
    times = []

    next_start = 0
    total_frames = len(x)

    while next_start + n_fft <= len(x_pad):
        orig_start = next_start - pad_len
        if 0 <= orig_start < total_frames:
            frame = x_pad[next_start:next_start + n_fft, :]
            left_levels.append(rms_dbfs(frame[:, 0]))
            right_levels.append(rms_dbfs(frame[:, 1]))
            times.append(orig_start / sr)
        next_start += hop

    left_levels = np.array(left_levels)
    right_levels = np.array(right_levels)

    print(f"总帧数: {len(left_levels)}")

    # 分别计算左右声道的最优阈值
    print("计算左声道阈值...")
    left_T = find_optimal_threshold(left_levels, target_c2, hyst_db, min_hold_frames)
    left_states = simulate_gate(left_levels, left_T, hyst_db, min_hold_frames)
    left_c2 = sum(1 for s in left_states if s == 'C2') / len(left_states)

    print("计算右声道阈值...")
    right_T = find_optimal_threshold(right_levels, target_c2, hyst_db, min_hold_frames)
    right_states = simulate_gate(right_levels, right_T, hyst_db, min_hold_frames)
    right_c2 = sum(1 for s in right_states if s == 'C2') / len(right_states)

    print(f"左声道: T={left_T:.2f} dBFS, C2={left_c2*100:.1f}%")
    print(f"右声道: T={right_T:.2f} dBFS, C2={right_c2*100:.1f}%")

    # 写入CSV
    print(f"写入: {out_csv}")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Frame',
            '音频秒数(秒)',
            '音频时间(分:秒)',
            'Left_dBFS',
            'Left_Channel',
            'Right_dBFS',
            'Right_Channel'
        ])

        for i, t in enumerate(times):
            writer.writerow([
                i + 1,
                f'{t:.3f}',
                format_time(t),
                f'{left_levels[i]:.2f}',
                left_states[i],
                f'{right_levels[i]:.2f}',
                right_states[i]
            ])

    print("完成")
    return 0


def main():
    parser = argparse.ArgumentParser(description='立体声状态分析器')
    parser.add_argument('-i', '--input', required=True, help='输入音频')
    parser.add_argument('-o', '--output', required=True, help='输出CSV')
    parser.add_argument('--target_c2', type=float, default=0.5)
    parser.add_argument('--hyst_db', type=float, default=3.0)
    parser.add_argument('--min_hold_ms', type=float, default=250.0)

    args = parser.parse_args()
    return analyze(args.input, args.output, args.target_c2, args.hyst_db, args.min_hold_ms)


if __name__ == '__main__':
    exit(main())
