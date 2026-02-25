#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并处理状态和输出音量
- 状态(C1/C2): 来自处理时的记录
- 音量(dBFS): 来自处理后的音频
"""

import argparse
import csv
import numpy as np
import soundfile as sf

EPS = 1e-12


def rms_dbfs(x_mono):
    r = np.sqrt(np.mean(x_mono * x_mono) + EPS)
    return float(20.0 * np.log10(r + EPS))


def format_time(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_csv', required=True, help='处理时的状态CSV')
    parser.add_argument('--audio', required=True, help='处理后的音频')
    parser.add_argument('-o', '--output', required=True, help='输出CSV')
    parser.add_argument('--n_fft', type=int, default=4096)
    parser.add_argument('--hop', type=int, default=2048)
    args = parser.parse_args()

    # 读取状态记录
    print(f"读取状态: {args.state_csv}")
    states = []
    with open(args.state_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            states.append(row['state'])

    # 读取处理后音频
    print(f"读取音频: {args.audio}")
    x, sr = sf.read(args.audio, dtype='float32')
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ch = x.shape[1]

    # 计算每帧dBFS
    pad_len = args.n_fft // 2
    x_pad = np.vstack([np.zeros((pad_len, ch)), x, np.zeros((pad_len, ch))])

    left_levels = []
    right_levels = []
    next_start = 0
    total = len(x)

    while next_start + args.n_fft <= len(x_pad):
        orig_start = next_start - pad_len
        if 0 <= orig_start < total:
            frame = x_pad[next_start:next_start + args.n_fft, :]
            left_levels.append(rms_dbfs(frame[:, 0]))
            if ch > 1:
                right_levels.append(rms_dbfs(frame[:, 1]))
        next_start += args.hop

    # 时间从hop/sr开始，每帧递增hop/sr
    frame_sec = args.hop / sr
    times = [(i + 1) * frame_sec for i in range(len(left_levels))]

    print(f"帧数: 状态={len(states)}, 音频={len(left_levels)}")

    # 写入合并CSV
    print(f"写入: {args.output}")
    n = min(len(states), len(left_levels))

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if ch > 1:
            writer.writerow([
                'Frame', '音频秒数(秒)', '音频时间(分:秒)',
                'Left_dBFS', 'Left_Channel',
                'Right_dBFS', 'Right_Channel'
            ])
            for i in range(n):
                writer.writerow([
                    i + 1,
                    f'{times[i]:.3f}',
                    format_time(times[i]),
                    f'{left_levels[i]:.2f}',
                    states[i],
                    f'{right_levels[i]:.2f}',
                    states[i]
                ])
        else:
            writer.writerow([
                'Frame', '音频秒数(秒)', '音频时间(分:秒)',
                'dBFS', 'Channel'
            ])
            for i in range(n):
                writer.writerow([
                    i + 1, f'{times[i]:.3f}', format_time(times[i]),
                    f'{left_levels[i]:.2f}', states[i]
                ])

    print("完成")


if __name__ == '__main__':
    main()
