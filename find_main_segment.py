import argparse, numpy as np, soundfile as sf

EPS = 1e-12

def win_rms_dbfs(frame_lr: np.ndarray) -> float:
    # frame_lr: [win, 2]
    # power average RMS: sqrt(mean((L^2+R^2)/2))
    p = (frame_lr[:, 0] * frame_lr[:, 0] + frame_lr[:, 1] * frame_lr[:, 1]) * 0.5
    r = np.sqrt(np.mean(p) + EPS)
    return float(20.0 * np.log10(r + EPS))

def find_segments(active: np.ndarray):
    # return list of (start_idx, end_idx_exclusive)
    segs = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--win_ms", type=float, default=100.0, help="窗口长度（ms）")
    ap.add_argument("--hop_ms", type=float, default=50.0, help="步长（ms）")
    ap.add_argument("--margin_db", type=float, default=15.0, help="高于噪声底多少dB算有音乐")
    ap.add_argument("--min_seg_sec", type=float, default=60.0, help="最短主段长度（秒）")
    ap.add_argument("--pad_sec", type=float, default=0.5, help="裁剪前后额外保留（秒）")
    args = ap.parse_args()

    with sf.SoundFile(args.input, "r") as f:
        sr = f.samplerate
        ch = f.channels
        n_total = f.frames
        dur = n_total / sr
        if ch != 2:
            raise ValueError(f"期望双声道，实际 {ch}")

        win = int(sr * args.win_ms / 1000.0)
        hop = int(sr * args.hop_ms / 1000.0)
        win_sec = win / sr

        # 流式提取 RMS(dBFS)
        levels = []
        times = []

        buf = np.zeros((0, ch), np.float32)
        base = 0
        next_start = 0

        block = sr * 10  # 每次读10秒
        while True:
            x = f.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            buf = np.vstack([buf, x])

            while True:
                rel = next_start - base
                if rel + win > len(buf):
                    break
                frame = buf[rel:rel+win, :]
                db = win_rms_dbfs(frame)
                levels.append(db)
                times.append(next_start / sr)
                next_start += hop

            # 保留最后 win 以支持重叠窗口
            keep_from = max(0, next_start - win)
            drop = keep_from - base
            if drop > 0:
                buf = buf[drop:, :]
                base += drop

        levels = np.array(levels, np.float32)
        times = np.array(times, np.float32)

    # 噪声底估计：取 p10（避免被短暂静音/断流影响）
    noise_floor = float(np.percentile(levels, 10))
    thr = noise_floor + args.margin_db

    active = levels >= thr
    segs = find_segments(active)

    if not segs:
        print("未找到主音乐段：请调低 margin_db 或检查音频是否几乎全静音。")
        return

    # 选"最长"的 active 段作为主音乐段
    best = None
    best_len = -1.0
    for i, j in segs:
        t0 = float(times[i])
        t1 = float(times[j-1] + win_sec)
        L = t1 - t0
        if L > best_len:
            best_len = L
            best = (t0, t1)

    t0, t1 = best
    if best_len < args.min_seg_sec:
        print(f"最长段只有 {best_len:.1f}s，小于 min_seg_sec={args.min_seg_sec}，建议调参。")
        return

    # 前后加 pad，并裁到 [0, dur]
    t0p = max(0.0, t0 - args.pad_sec)
    t1p = min(dur, t1 + args.pad_sec)

    print("==== 自动检测结果 ====")
    print(f"总时长: {dur:.2f}s  采样率: {sr}Hz  声道: {ch}")
    print(f"噪声底(p10): {noise_floor:.1f} dBFS")
    print(f"活动阈值: {thr:.1f} dBFS (noise_floor + {args.margin_db}dB)")
    print(f"主音乐段(未加pad): start={t0:.3f}s  end={t1:.3f}s  len={best_len:.1f}s")
    print(f"建议裁剪(加pad):  start={t0p:.3f}s  end={t1p:.3f}s  len={(t1p-t0p):.1f}s")

if __name__ == "__main__":
    main()
