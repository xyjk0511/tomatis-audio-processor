"""
Tomatis 音频处理器 - Gate 控制的 C1/C2 倾斜滤波器 (带 Crossfade)

基于 process_tomatis.py，添加 crossfade 平滑过渡功能
- 保持与 Matlab 版本的高匹配度
- 添加 C1/C2 状态切换时的平滑过渡

作者: DSP 分析工具
日期: 2026-01-20
版本: 2.0 (添加 crossfade)
"""

import argparse
import numpy as np
import soundfile as sf
import csv

EPS = 1e-12
PEAK_LIMIT = 0.999


def rms_dbfs(x_mono: np.ndarray) -> float:
    """计算单声道帧的 RMS dBFS"""
    r = np.sqrt(np.mean(x_mono * x_mono) + EPS)
    return float(20.0 * np.log10(r + EPS))


def gate_ui_to_dbfs(gate_ui: float, gate_scale: float = 1.0, gate_offset: float = -100.0) -> float:
    """将 UI gate 值 (0-100) 转换为 dBFS 阈值"""
    return gate_scale * gate_ui + gate_offset


def db_to_lin(db: np.ndarray) -> np.ndarray:
    """dB 转线性增益"""
    return (10.0 ** (np.asarray(db) / 20.0)).astype(np.float32)


def build_tilt_gain_db(freqs, fc, slope_db_per_oct, low_gain_db, high_gain_db):
    """生成以 fc 为 0dB 支点的倾斜增益曲线"""
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


def process(
    in_path,
    out_path,
    gate_ui=50,
    gate_scale=1.0,
    gate_offset=-100,
    hysteresis_db=3.0,
    fc=1000.0,
    slope=12.0,
    c1_low=+15.0, c1_high=-15.0,
    c2_low=-15.0, c2_high=+15.0,
    up_delay_ms=250.0,
    xfade_ms=0.0,  # 新增：crossfade 过渡时间
    n_fft=4096,
    hop=2048,
    state_csv_path=None,
):
    """
    主处理函数：对输入音频应用 gate 控制的 C1/C2 倾斜滤波 (带可选 crossfade)
    
    新增参数:
        xfade_ms: Crossfade 过渡时间（ms），0 表示硬切换
    """
    print("=" * 70)
    print("Tomatis 音频处理器 (Crossfade 版)")
    print("=" * 70)
    print(f"\n输入文件: {in_path}")
    print(f"输出文件: {out_path}")
    print(f"\n参数配置:")
    print(f"  Gate UI: {gate_ui} (阈值: {gate_ui_to_dbfs(gate_ui, gate_scale, gate_offset):.1f} dBFS)")
    print(f"  回差: {hysteresis_db} dB")
    print(f"  上行延迟: {up_delay_ms} ms")
    print(f"  Crossfade: {xfade_ms} ms")
    print(f"  中心频率: {fc} Hz")
    print(f"  坡度: {slope} dB/octave")
    print(f"  C1 增益: 低频{c1_low:+.1f}dB, 高频{c1_high:+.1f}dB")
    print(f"  C2 增益: 低频{c2_low:+.1f}dB, 高频{c2_high:+.1f}dB")
    print(f"  FFT 参数: n_fft={n_fft}, hop={hop}")
    print()

    # --- 读输入 ---
    print("正在读取输入文件...")
    with sf.SoundFile(in_path, 'r') as fin:
        sr = fin.samplerate
        ch = fin.channels
        total_frames = fin.frames
        
        print(f"✓ 采样率: {sr} Hz")
        print(f"✓ 声道数: {ch}")
        print(f"✓ 总长度: {total_frames} 采样点 ({total_frames/sr:.2f} 秒)")
        
        if sr != 48000:
            raise ValueError(f"期望 48kHz，实际 {sr} Hz")
        if ch != 2:
            raise ValueError(f"期望双声道，实际 {ch} 声道")
        print()

        # 输出文件
        print("正在创建输出文件...")
        try:
            fout = sf.SoundFile(out_path, 'w', samplerate=sr, channels=ch, format='FLAC', subtype='PCM_24')
            out_is_flac = True
            print("✓ 输出格式: FLAC 24-bit")
        except Exception as e:
            print(f"⚠ FLAC 写入失败: {e}")
            wav_path = out_path.replace(".flac", ".wav")
            fout = sf.SoundFile(wav_path, 'w', samplerate=sr, channels=ch, format='WAV', subtype='PCM_24')
            out_is_flac = False
            print(f"✓ 输出格式: WAV 24-bit")
        print()

        # --- 预计算增益曲线 ---
        print("正在计算增益曲线...")
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        g1_db = build_tilt_gain_db(freqs, fc, slope, c1_low, c1_high)
        g2_db = build_tilt_gain_db(freqs, fc, slope, c2_low, c2_high)
        g1 = db_to_lin(g1_db)
        g2 = db_to_lin(g2_db)
        print(f"✓ C1/C2 增益曲线: {len(g1)} 频率 bins")
        print()

        # --- 窗与 OLA ---
        win = np.hanning(n_fft).astype(np.float32)
        win2 = (win * win).astype(np.float32)
        
        # --- Padding ---
        pad = n_fft // 2
        pad_end = (hop - ((total_frames - n_fft) % hop)) % hop
        print(f"边界处理: padding_start = {pad}, padding_end = {pad_end} 采样点")
        print()

        # --- 门限 ---
        T = gate_ui_to_dbfs(gate_ui, gate_scale, gate_offset)
        Ton = T + hysteresis_db / 2.0
        Toff = T - hysteresis_db / 2.0
        up_delay_samples = int(sr * up_delay_ms / 1000.0)
        
        # --- Crossfade 参数 ---
        frame_duration_ms = hop / sr * 1000.0
        xfade_frames = max(1, int(np.ceil(xfade_ms / frame_duration_ms))) if xfade_ms > 0 else 0
        alpha_step = 1.0 / xfade_frames if xfade_frames > 0 else 1.0
        
        print("门控参数:")
        print(f"  基础阈值: {T:.1f} dBFS")
        print(f"  上行阈值 (C1→C2): {Ton:.1f} dBFS")
        print(f"  下行阈值 (C2→C1): {Toff:.1f} dBFS")
        print(f"  上行延迟: {up_delay_samples} 采样点 ({up_delay_ms} ms)")
        if xfade_ms > 0:
            print(f"  Crossfade: {xfade_frames} 帧 ({xfade_ms} ms)")
        print()

        # --- 状态机 ---
        state = 1  # 1=C1, 2=C2
        pending_c2_at = None
        
        # --- Alpha 控制 (0=C1, 1=C2) ---
        current_alpha = 0.0
        target_alpha = 0.0

        # --- CSV 记录 ---
        csv_writer = None
        csv_file = None
        if state_csv_path:
            csv_file = open(state_csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame_idx", "time_sec", "level_dbfs", "state", "alpha"])
            print(f"✓ 状态记录: {state_csv_path}")
            print()

        # --- 流式处理缓冲 ---
        in_buf = np.zeros((pad, ch), dtype=np.float32)
        in_base = -pad
        next_start = -pad

        out_buf = np.zeros((0, ch), dtype=np.float32)
        w_buf = np.zeros((0,), dtype=np.float32)
        out_base = -pad

        frame_idx = 0
        c1_frames = 0
        c2_frames = 0

        def ensure_out(end_pos):
            nonlocal out_buf, w_buf, out_base
            need = end_pos - out_base
            if need <= len(w_buf):
                return
            grow = need - len(w_buf)
            out_buf = np.vstack([out_buf, np.zeros((grow, ch), np.float32)])
            w_buf = np.concatenate([w_buf, np.zeros((grow,), np.float32)])
        
        def write_clamped(y_chunk: np.ndarray, abs_start: int):
            """只写 [0, total_frames) 范围内的样本"""
            abs_end = abs_start + len(y_chunk)
            s = max(0, abs_start)
            e = min(total_frames, abs_end)
            if e <= s:
                return
            cs = s - abs_start
            ce = e - abs_start
            out_chunk = y_chunk[cs:ce]
            peak = np.max(np.abs(out_chunk))
            if peak > PEAK_LIMIT:
                out_chunk = out_chunk * (PEAK_LIMIT / peak)
            fout.write(out_chunk)

        def process_available_frames():
            nonlocal in_buf, in_base, next_start, state, pending_c2_at
            nonlocal c1_frames, c2_frames, frame_idx
            nonlocal out_buf, w_buf, out_base
            nonlocal current_alpha, target_alpha

            while True:
                rel = next_start - in_base
                if rel + n_fft > len(in_buf):
                    break

                frame = in_buf[rel:rel+n_fft, :]
                mono = np.sqrt(np.mean(frame**2, axis=1))
                level = rms_dbfs(mono)

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

                # 更新目标 alpha
                target_alpha = 0.0 if state == 1 else 1.0

                # 平滑 alpha（crossfade）
                if xfade_frames > 0:
                    diff = target_alpha - current_alpha
                    if abs(diff) <= alpha_step:
                        current_alpha = target_alpha
                    else:
                        current_alpha += alpha_step * np.sign(diff)
                else:
                    current_alpha = target_alpha

                if state == 1:
                    c1_frames += 1
                else:
                    c2_frames += 1

                # 混合增益（在 dB 域）
                if xfade_ms > 0 and 0 < current_alpha < 1:
                    mixed_gain_db = (1 - current_alpha) * g1_db + current_alpha * g2_db
                    gain = db_to_lin(mixed_gain_db)
                else:
                    gain = g1 if current_alpha < 0.5 else g2

                # 频域处理
                y = np.zeros_like(frame, dtype=np.float32)
                for c in range(ch):
                    X = np.fft.rfft(frame[:, c] * win)
                    X *= gain
                    y[:, c] = np.fft.irfft(X, n=n_fft).astype(np.float32) * win

                # Overlap-Add
                start = next_start
                end = start + n_fft
                ensure_out(end)

                orel = start - out_base
                out_buf[orel:orel+n_fft, :] += y
                w_buf[orel:orel+n_fft] += win2

                # CSV 记录
                if csv_writer and 0 <= start < total_frames:
                    csv_writer.writerow([frame_idx, start / sr, f"{level:.2f}", 
                                        "C1" if state == 1 else "C2", f"{current_alpha:.3f}"])

                frame_idx += 1
                next_start += hop

                if frame_idx % 1000 == 0:
                    actual_progress = max(0, next_start) / total_frames * 100
                    print(f"\r处理进度: {actual_progress:.1f}% ({frame_idx} 帧)", end='', flush=True)

                # 定期写出
                safe = (next_start - out_base) - n_fft
                if safe >= 48000 * 5:
                    n = safe
                    y_out = out_buf[:n, :] / (w_buf[:n, None] + EPS)
                    write_clamped(y_out, out_base)
                    out_base += n
                    out_buf = out_buf[n:, :]
                    w_buf = w_buf[n:]

        # 主处理循环
        block = 48000 * 10
        print("开始处理...")
        print()
        
        while True:
            x = fin.read(block, dtype='float32', always_2d=True)
            if len(x) == 0:
                break
            in_buf = np.vstack([in_buf, x])
            process_available_frames()
            keep = max(0, len(in_buf) - n_fft)
            if keep > 0:
                in_buf = in_buf[keep:, :]
                in_base += keep

        # 收尾
        if pad_end > 0:
            in_buf = np.vstack([in_buf, np.zeros((pad_end, ch), dtype=np.float32)])
        process_available_frames()

        if len(w_buf) > 0:
            y_out = out_buf / (w_buf[:, None] + EPS)
            write_clamped(y_out, out_base)

        fout.close()
        if csv_file:
            csv_file.close()

        print("\r" + " " * 80 + "\r", end='')
        print()
        print("=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"\n统计信息:")
        print(f"  总帧数: {frame_idx}")
        print(f"  C1 帧数: {c1_frames} ({c1_frames/frame_idx*100:.1f}%)")
        print(f"  C2 帧数: {c2_frames} ({c2_frames/frame_idx*100:.1f}%)")
        if xfade_ms > 0:
            print(f"  Crossfade: {xfade_ms} ms ({xfade_frames} 帧)")
        print(f"\n输出文件: {out_path if out_is_flac else wav_path}")
        
        if state_csv_path:
            print(f"状态记录: {state_csv_path}")
        
        print()


def main():
    ap = argparse.ArgumentParser(
        description="Tomatis 音频处理器 - Gate 控制的 C1/C2 倾斜滤波器 (带 Crossfade)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ap.add_argument("-i", "--input", required=True, help="输入 FLAC 文件")
    ap.add_argument("-o", "--output", required=True, help="输出 FLAC 文件")
    
    # Gate 参数
    ap.add_argument("--gate_ui", type=float, default=50, help="Gate UI 值 (0-100)")
    ap.add_argument("--gate_scale", type=float, default=1.0, help="Gate 缩放系数")
    ap.add_argument("--gate_offset", type=float, default=-100, help="Gate 偏移量")
    ap.add_argument("--hyst_db", type=float, default=3.0, help="回差（dB）")
    ap.add_argument("--up_delay_ms", type=float, default=250.0, help="C1→C2 上行延迟（ms）")
    ap.add_argument("--xfade_ms", type=float, default=0.0, help="Crossfade 过渡时间（ms），0=硬切换")
    
    # 滤波器参数
    ap.add_argument("--fc", type=float, default=1000.0, help="中心频率（Hz）")
    ap.add_argument("--slope", type=float, default=12.0, help="坡度（dB/octave）")
    ap.add_argument("--c1_low", type=float, default=15.0, help="C1 低频增益（dB）")
    ap.add_argument("--c1_high", type=float, default=-15.0, help="C1 高频增益（dB）")
    ap.add_argument("--c2_low", type=float, default=-15.0, help="C2 低频增益（dB）")
    ap.add_argument("--c2_high", type=float, default=15.0, help="C2 高频增益（dB）")
    
    # FFT 参数
    ap.add_argument("--n_fft", type=int, default=4096, help="FFT 窗长")
    ap.add_argument("--hop", type=int, default=2048, help="跳步长度")
    
    ap.add_argument("--state_csv", default=None, help="输出状态 CSV 文件路径")
    
    args = ap.parse_args()

    try:
        process(
            args.input, args.output,
            gate_ui=args.gate_ui,
            gate_scale=args.gate_scale,
            gate_offset=args.gate_offset,
            hysteresis_db=args.hyst_db,
            fc=args.fc,
            slope=args.slope,
            c1_low=args.c1_low, c1_high=args.c1_high,
            c2_low=args.c2_low, c2_high=args.c2_high,
            up_delay_ms=args.up_delay_ms,
            xfade_ms=args.xfade_ms,
            n_fft=args.n_fft,
            hop=args.hop,
            state_csv_path=args.state_csv
        )
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
