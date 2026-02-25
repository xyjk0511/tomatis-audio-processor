"""
Tomatis 音频处理器 - Gate 控制的 C1/C2 倾斜滤波器

实现说明：
- C1: 低频增强(+5dB)、高频衰减(-5dB) - 用于安静段落
- C2: 低频衰减(-5dB)、高频增强(+5dB) - 用于响亮段落
- Gate: 基于 RMS dBFS 的门控切换，带回差和上行延迟
- 处理方式: 短时 FFT + 频域增益 + Overlap-Add (OLA)

重要技术说明：
1. dBFS 定义：
   - 本程序中所有 dB 均指 dBFS（满幅 0 dBFS）
   - 正常音频电平为负值（例如 -60 到 -10 dBFS）
   - 电平测量：RMS dBFS，多声道用能量平均（power average）

2. Gate 映射：
   - 通用映射: T_dBFS = gate_scale * gate_ui + gate_offset
   - 默认: gate_scale=1.0, gate_offset=-100
   - gate_ui=50 → T_dBFS = -50 dBFS

3. 倾斜增益曲线：
   - 中心频率 fc 处增益为 0 dB
   - 坡度 slope 决定每 octave 的增益变化
   - 平台开始频率: f_hi = fc * 2^(|G_hi|/slope)

4. OLA 归一化：
   - 分析窗 = 合成窗 (Hann)
   - 输出归一化: y[n] = Σy_k[n] / (Σw²[n] + ε)

作者: DSP 分析工具
日期: 2026-01-18
版本: 1.3 (添加 pad/trim 边界处理，消除开头/结尾掉底)
"""

import argparse
import numpy as np
import soundfile as sf
import csv

EPS = 1e-12
PEAK_LIMIT = 0.999

def rms_dbfs(x_mono: np.ndarray) -> float:
    """
    计算单声道帧的 RMS dBFS
    
    注意: 如果输入是多声道，应该先做能量平均（power average）:
        x_mono = sqrt(mean(L^2 + R^2) / 2)
    而不是波形平均 (L+R)/2，后者在反相时会严重低估能量
    """
    r = np.sqrt(np.mean(x_mono * x_mono) + EPS)
    return float(20.0 * np.log10(r + EPS))

def gate_ui_to_dbfs(gate_ui: float, gate_scale: float = 1.0, gate_offset: float = -100.0) -> float:
    """
    将 UI gate 值 (0-100) 转换为 dBFS 阈值
    
    通用映射: T_dBFS = gate_scale * gate_ui + gate_offset
    - 默认: gate_scale=1.0, gate_offset=-100
    - gate_ui=50 → T_dBFS = 1.0*50 + (-100) = -50 dBFS
    
    校准方法:
    1. 在设备上测试多个 gate_ui 值，记录实际切换行为
    2. 用 analyze_dbfs.py 找到对应的 dBFS 阈值
    3. 拟合线性关系得到 gate_scale 和 gate_offset
    
    注意:
    - 本程序中所有 dB 均指 dBFS（满幅 0 dBFS）
    - 正常音频电平为负值（例如 -60 dBFS 到 -10 dBFS）
    - 设备 UI 的 0-100 是抽象量，经此映射到 dBFS
    
    参数:
        gate_ui: UI 标尺值 (0-100)
        gate_scale: 缩放系数（默认 1.0）
        gate_offset: 偏移量（默认 -100）
    
    返回:
        dBFS 阈值
    """
    return gate_scale * gate_ui + gate_offset

def gate_ui_to_dbfs_log_percent(gate_ui: float, dynamic_range: float = 80.0) -> float:
    """
    对数百分比门控换算

    设备使用对数百分比作为门控单位，分母是声音的最大强度(0 dBFS)。
    gate_ui 表示在对数刻度(dB)上的百分比位置。

    公式: T_dBFS = -dynamic_range + dynamic_range * gate_ui / 100

    示例 (dynamic_range=80):
        gate_ui = 0   -> -80 dBFS (最敏感，很小声就切换)
        gate_ui = 50  -> -40 dBFS (中间)
        gate_ui = 100 -> 0 dBFS (最不敏感，要很大声才切换)

    参数:
        gate_ui: UI 标尺值 (0-100)
        dynamic_range: 动态范围 (dB)，默认 80dB

    返回:
        dBFS 阈值
    """
    return -dynamic_range + dynamic_range * gate_ui / 100.0

def db_to_lin(db: np.ndarray) -> np.ndarray:
    """dB 转线性增益"""
    return (10.0 ** (db / 20.0)).astype(np.float32)

def build_tilt_gain_db(freqs, fc, slope_db_per_oct, low_gain_db, high_gain_db):
    """
    生成"以 fc 为 0dB 支点"的倾斜增益曲线
    
    实现原理: 左右两侧分别爬坡再平台钳位
    - 低频侧 (f < fc): 从 0dB 朝 low_gain_db 爬坡，到达后平台
    - 高频侧 (f > fc): 从 0dB 朝 high_gain_db 爬坡，到达后平台
    
    参数:
        freqs: FFT 频率 bins
        fc: 中心频率（Hz），在此处增益为 0 dB
        slope_db_per_oct: 坡度（dB/octave），例如 6, 12, 18
        low_gain_db: 低频平台增益（dB），可正可负
        high_gain_db: 高频平台增益（dB），可正可负
    
    返回:
        增益曲线（dB）
    
    公式说明:
        x = log2(f / fc)  # 距离中心频率的倍频程距离
        
        低频侧 (x < 0):
            d_low = slope * |x|  # 爬坡距离
            g = sign(low_gain_db) * min(d_low, |low_gain_db|)
        
        高频侧 (x > 0):
            d_hi = slope * x
            g = sign(high_gain_db) * min(d_hi, |high_gain_db|)
    
    平台开始频率:
        高频平台: f_hi = fc * 2^(|G_hi| / slope)
        低频平台: f_lo = fc * 2^(-|G_lo| / slope)
    
    注意: 不要用单一 clip(low, high)，当 low_db > high_db 时会逻辑崩溃
    """
    f = np.maximum(freqs, 1.0)  # 避免 log2(0)
    x = np.log2(f / fc).astype(np.float32)  # <0 低频, >0 高频
    g = np.zeros_like(x, dtype=np.float32)

    # 低频侧 (f < fc): 爬坡距离随 fc/f 增大
    d_low = slope_db_per_oct * np.maximum(0.0, -x)  # -x > 0 for x < 0
    g_low = np.sign(low_gain_db) * np.minimum(d_low, abs(low_gain_db))
    g[x < 0] = g_low[x < 0]

    # 高频侧 (f > fc): 爬坡距离随 f/fc 增大
    d_hi = slope_db_per_oct * np.maximum(0.0, x)  # x > 0
    g_hi = np.sign(high_gain_db) * np.minimum(d_hi, abs(high_gain_db))
    g[x > 0] = g_hi[x > 0]

    return g

def process(
    in_path,
    out_path,
    gate_ui=50,
    gate_mode="log_percent",
    dynamic_range=80.0,
    gate_scale=1.0,
    gate_offset=-100,
    hysteresis_db=3.0,
    fc=1000.0,
    slope=12.0,
    c1_low=+15.0, c1_high=-15.0,
    c2_low=-15.0, c2_high=+15.0,
    up_delay_ms=250.0,
    n_fft=4096,
    hop=2048,
    state_csv_path=None,
    output_gain_db=0.0,
):
    """
    主处理函数：对输入音频应用 gate 控制的 C1/C2 倾斜滤波
    
    参数:
        in_path: 输入 FLAC 文件路径
        out_path: 输出 FLAC 文件路径
        gate_ui: Gate UI 值 (0-100)
        gate_scale: Gate 缩放系数（默认 1.0）
        gate_offset: Gate 偏移量（默认 -100）
        hysteresis_db: 回差（dB），避免抖动
        fc: 中心频率（Hz）
        slope: 坡度（dB/octave）
        c1_low/c1_high: C1 的低频/高频平台增益
        c2_low/c2_high: C2 的低频/高频平台增益
        up_delay_ms: C1→C2 上行延迟（ms）
        n_fft: FFT 窗长
        hop: 跳步长度
        state_csv_path: 可选，输出状态 CSV
        
    注意:
        - 切换时刻精度约为 hop/sr 秒（帧边界）
        - 30分钟双声道 float32 + OLA 缓冲需 1-2GB 内存
    """
    print("=" * 70)
    print("Tomatis 音频处理器")
    print("=" * 70)
    print(f"\n输入文件: {in_path}")
    print(f"输出文件: {out_path}")
    print(f"\n参数配置:")
    if gate_mode == "log_percent":
        threshold_preview = gate_ui_to_dbfs_log_percent(gate_ui, dynamic_range)
        print(f"  Gate UI: {gate_ui} (模式: 对数百分比, 阈值: {threshold_preview:.1f} dBFS)")
    else:
        threshold_preview = gate_ui_to_dbfs(gate_ui, gate_scale, gate_offset)
        print(f"  Gate UI: {gate_ui} (模式: 线性, 阈值: {threshold_preview:.1f} dBFS)")
    print(f"  回差: {hysteresis_db} dB")
    print(f"  上行延迟: {up_delay_ms} ms")
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
        
        print(f"[OK] 采样率: {sr} Hz")
        print(f"[OK] 声道数: {ch}")
        print(f"[OK] 总长度: {total_frames} 采样点 ({total_frames/sr:.2f} 秒)")
        
        if sr != 48000:
            raise ValueError(f"期望 48kHz，实际 {sr} Hz")
        if ch != 2:
            raise ValueError(f"期望双声道，实际 {ch} 声道")
        print()

        # 输出：优先直接写 FLAC
        print("正在创建输出文件...")
        try:
            fout = sf.SoundFile(out_path, 'w', samplerate=sr, channels=ch, format='FLAC', subtype='PCM_24')
            out_is_flac = True
            print("[OK] 输出格式: FLAC 24-bit")
        except Exception as e:
            print(f"[WARN] FLAC 写入失败: {e}")
            wav_path = out_path.replace(".flac", ".wav")
            fout = sf.SoundFile(wav_path, 'w', samplerate=sr, channels=ch, format='WAV', subtype='PCM_24')
            out_is_flac = False
            print(f"[OK] 输出格式: WAV 24-bit (稍后需转换为 FLAC)")
        print()

        # --- 预计算增益曲线（频域） ---
        print("正在计算增益曲线...")
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        g1_db = build_tilt_gain_db(freqs, fc, slope, c1_low, c1_high)
        g2_db = build_tilt_gain_db(freqs, fc, slope, c2_low, c2_high)
        g1 = db_to_lin(g1_db)
        g2 = db_to_lin(g2_db)
        print(f"[OK] C1 增益曲线: {len(g1)} 频率 bins")
        print(f"[OK] C2 增益曲线: {len(g2)} 频率 bins")
        print()

        # --- 窗与 OLA 归一化 ---
        win = np.hanning(n_fft).astype(np.float32)
        win2 = (win * win).astype(np.float32)
        
        # --- Padding: 消除边界掉底 ---
        pad = n_fft // 2  # 半窗 padding，等价"居中帧"
        # 尾部补零：确保最后几帧被覆盖
        pad_end = (hop - ((total_frames - n_fft) % hop)) % hop
        print(f"边界处理: padding_start = {pad}, padding_end = {pad_end} 采样点")
        print()

        # --- 门限 ---
        if gate_mode == "log_percent":
            T = gate_ui_to_dbfs_log_percent(gate_ui, dynamic_range)
            mode_str = f"对数百分比 (动态范围={dynamic_range}dB)"
        else:
            T = gate_ui_to_dbfs(gate_ui, gate_scale, gate_offset)
            mode_str = f"线性 (scale={gate_scale}, offset={gate_offset})"
        Ton = T + hysteresis_db / 2.0
        Toff = T - hysteresis_db / 2.0
        up_delay_samples = int(sr * up_delay_ms / 1000.0)

        print("门控参数:")
        print(f"  模式: {mode_str}")
        print(f"  基础阈值: {T:.1f} dBFS")
        print(f"  上行阈值 (C1→C2): {Ton:.1f} dBFS")
        print(f"  下行阈值 (C2→C1): {Toff:.1f} dBFS")
        print(f"  上行延迟: {up_delay_samples} 采样点 ({up_delay_ms} ms)")
        print()

        # --- 状态机 ---
        state = 1  # 1=C1, 2=C2
        pending_c2_at = None  # 计划切到 C2 的绝对样点位置

        # --- 可选：记录每帧状态 ---
        csv_writer = None
        csv_file = None
        if state_csv_path:
            csv_file = open(state_csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame_idx", "time_sec", "level_dbfs", "state"])
            print(f"[OK] 状态记录: {state_csv_path}")
            print()

        # --- 流式处理缓冲（带 padding）---
        in_buf = np.zeros((pad, ch), dtype=np.float32)  # 前置 pad
        in_base = -pad                                   # 输入缓冲对应的绝对样点
        next_start = -pad                                # 第一帧从 -pad 开始

        out_buf = np.zeros((0, ch), dtype=np.float32)
        w_buf = np.zeros((0,), dtype=np.float32)
        out_base = -pad                                  # 输出缓冲也用同一绝对坐标

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
            """
            只把 y_chunk 中落在 [0, total_frames) 的部分写出
            abs_start 是该 chunk 在原始音频坐标系里的起点（可为负，因为做了 pad）
            包含峰值保护：限幅到 0.999 避免削波
            """
            abs_end = abs_start + len(y_chunk)

            s = max(0, abs_start)
            e = min(total_frames, abs_end)
            if e <= s:
                return

            cs = s - abs_start
            ce = e - abs_start
            out_chunk = y_chunk[cs:ce]

            # 应用输出增益补偿
            if output_gain_db != 0.0:
                out_chunk = out_chunk * (10.0 ** (output_gain_db / 20.0))

            # 峰值保护：限幅到 0.999 避免削波
            peak = np.max(np.abs(out_chunk))
            if peak > PEAK_LIMIT:
                out_chunk = out_chunk * (PEAK_LIMIT / peak)

            fout.write(out_chunk)

        def process_available_frames():
            nonlocal in_buf, in_base, next_start, state, pending_c2_at
            nonlocal c1_frames, c2_frames, frame_idx
            nonlocal out_buf, w_buf, out_base

            while True:
                rel = next_start - in_base
                if rel + n_fft > len(in_buf):
                    break

                frame = in_buf[rel:rel+n_fft, :]
                mono = np.sqrt(np.mean(frame**2, axis=1))
                level = rms_dbfs(mono)

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

                if state == 1:
                    c1_frames += 1
                else:
                    c2_frames += 1

                gain = g1 if state == 1 else g2

                y = np.zeros_like(frame, dtype=np.float32)
                for c in range(ch):
                    X = np.fft.rfft(frame[:, c] * win)
                    X *= gain
                    y[:, c] = np.fft.irfft(X, n=n_fft).astype(np.float32) * win

                start = next_start
                end = start + n_fft
                ensure_out(end)

                orel = start - out_base
                out_buf[orel:orel+n_fft, :] += y
                w_buf[orel:orel+n_fft] += win2

                if csv_writer and 0 <= start < total_frames:
                    csv_writer.writerow([frame_idx, start / sr, level, "C1" if state == 1 else "C2"])

                frame_idx += 1
                next_start += hop

                if frame_idx % 1000 == 0:
                    actual_progress = max(0, next_start) / total_frames * 100
                    print(f"\r处理进度: {actual_progress:.1f}% ({frame_idx} 帧"
, end='', flush=True)

                safe = (next_start - out_base) - n_fft
                if safe >= 48000 * 5:
                    n = safe
                    y_out = out_buf[:n, :] / (w_buf[:n, None] + EPS)
                    write_clamped(y_out, out_base)
                    out_base += n
                    out_buf = out_buf[n:, :]
                    w_buf = w_buf[n:]

        # 每次读一块
        block = 48000 * 10  # 10秒
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

        # 收尾：把剩余输出写完（使用裁剪写出）
        if pad_end > 0:
            in_buf = np.vstack([in_buf, np.zeros((pad_end, ch), dtype=np.float32)])
        process_available_frames()

        if len(w_buf) > 0:
            y_out = out_buf / (w_buf[:, None] + EPS)
            write_clamped(y_out, out_base)

        fout.close()
        if csv_file:
            csv_file.close()

        print("\r" + " " * 80 + "\r", end='')  # 清除进度行
        print()
        print("=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"\n统计信息:")
        print(f"  总帧数: {frame_idx}")
        print(f"  C1 帧数: {c1_frames} ({c1_frames/frame_idx*100:.1f}%)")
        print(f"  C2 帧数: {c2_frames} ({c2_frames/frame_idx*100:.1f}%)")
        print(f"\n输出文件: {out_path if out_is_flac else wav_path}")
        print(f"  输出长度: {total_frames} 采样点 (与输入一致)")
        
        if not out_is_flac:
            print("\n[WARN] 注意: 已输出 WAV 格式（因 FLAC 写入失败）")
            print("请使用以下命令转换为 FLAC:")
            print(f'ffmpeg -y -i "{wav_path}" -c:a flac -compression_level 8 "{out_path}"')
        
        if state_csv_path:
            print(f"\n状态记录: {state_csv_path}")
        
        print()

def main():
    ap = argparse.ArgumentParser(
        description="Tomatis 音频处理器 - Gate 控制的 C1/C2 倾斜滤波器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    ap.add_argument("-i", "--input", required=True, help="输入 FLAC 文件")
    ap.add_argument("-o", "--output", required=True, help="输出 FLAC 文件")
    
    # Gate 参数
    ap.add_argument("--gate_ui", type=float, default=50, help="Gate UI 值 (0-100)")
    ap.add_argument("--gate_mode", choices=["linear", "log_percent"], default="log_percent",
                    help="门控换算模式: linear=线性公式, log_percent=对数百分比(推荐)")
    ap.add_argument("--dynamic_range", type=float, default=80.0, help="动态范围(dB)，用于log_percent模式")
    ap.add_argument("--gate_scale", type=float, default=1.0, help="Gate 缩放系数(linear模式)")
    ap.add_argument("--gate_offset", type=float, default=-100, help="Gate 偏移量(linear模式)")
    ap.add_argument("--hyst_db", type=float, default=3.0, help="回差（dB）")
    ap.add_argument("--up_delay_ms", type=float, default=250.0, help="C1→C2 上行延迟（ms）")
    
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
    
    # 可选输出
    ap.add_argument("--state_csv", default=None, help="输出状态 CSV 文件路径")
    ap.add_argument("--output_gain_db", type=float, default=0.0, help="输出增益补偿（dB）")

    args = ap.parse_args()

    try:
        process(
            args.input, args.output,
            gate_ui=args.gate_ui,
            gate_mode=args.gate_mode,
            dynamic_range=args.dynamic_range,
            gate_scale=args.gate_scale,
            gate_offset=args.gate_offset,
            hysteresis_db=args.hyst_db,
            fc=args.fc,
            slope=args.slope,
            c1_low=args.c1_low, c1_high=args.c1_high,
            c2_low=args.c2_low, c2_high=args.c2_high,
            up_delay_ms=args.up_delay_ms,
            n_fft=args.n_fft,
            hop=args.hop,
            state_csv_path=args.state_csv,
            output_gain_db=args.output_gain_db
        )
    except Exception as e:
        print(f"\n[ERR] 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
