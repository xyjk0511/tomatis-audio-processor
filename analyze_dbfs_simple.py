"""
简化的 dBFS 分析工具
用于确定合适的 gate 阈值

使用方法:
    python analyze_dbfs_simple.py -i "D MNF.flac"

输出百分位数，用于选择 gate_ui:
    - p50 (中位数): 适合作为基准阈值
    - p10/p90: 了解动态范围
"""

import argparse
import numpy as np
import soundfile as sf

EPS = 1e-12

def rms_dbfs(x):
    """计算 RMS dBFS"""
    r = np.sqrt(np.mean(x * x) + EPS)
    return 20 * np.log10(r + EPS)

def main():
    ap = argparse.ArgumentParser(
        description="分析音频的 RMS dBFS 分布，用于选择 gate 阈值"
    )
    ap.add_argument("-i", "--input", required=True, help="输入音频文件")
    ap.add_argument("--n_fft", type=int, default=4096, help="帧长")
    ap.add_argument("--hop", type=int, default=2048, help="跳步")
    args = ap.parse_args()

    print(f"正在分析: {args.input}")
    print()

    # 读取音频
    x, sr = sf.read(args.input, dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    
    print(f"采样率: {sr} Hz")
    print(f"总长度: {len(mono)} 采样点 ({len(mono)/sr:.2f} 秒)")
    print()

    # 计算每帧的 RMS dBFS
    n = args.n_fft
    hop = args.hop
    levels = []
    
    for s in range(0, len(mono) - n + 1, hop):
        frame = mono[s:s + n]
        levels.append(rms_dbfs(frame))
    
    levels = np.array(levels, dtype=np.float32)

    # 统计百分位数
    p1, p10, p50, p90, p99 = np.percentile(levels, [1, 10, 50, 90, 99])
    
    print("=" * 60)
    print("RMS dBFS 分析结果")
    print("=" * 60)
    print(f"\n总帧数: {len(levels)}")
    print(f"\n百分位数:")
    print(f"  p1  (最弱 1%):  {p1:6.1f} dBFS")
    print(f"  p10 (较弱 10%): {p10:6.1f} dBFS")
    print(f"  p50 (中位数):   {p50:6.1f} dBFS")
    print(f"  p90 (较强 10%): {p90:6.1f} dBFS")
    print(f"  p99 (最强 1%):  {p99:6.1f} dBFS")
    
    print(f"\n动态范围: {p99 - p1:.1f} dB")
    
    # 给出 gate_ui 建议
    print("\n" + "=" * 60)
    print("Gate 阈值建议")
    print("=" * 60)
    print("\n假设 gate_offset = -100 (默认)，则:")
    print(f"  gate_ui = T_dBFS - gate_offset = T_dBFS - (-100) = T_dBFS + 100")
    print()
    
    # 计算建议值
    gate_offset = -100
    suggestions = [
        ("p30 (30% 为 C2)", np.percentile(levels, 30)),
        ("p50 (50% 为 C2)", p50),
        ("p70 (70% 为 C2)", np.percentile(levels, 70)),
    ]
    
    print("建议的 gate_ui 值:")
    for desc, t_dbfs in suggestions:
        gate_ui = t_dbfs - gate_offset
        print(f"  {desc:20} → T = {t_dbfs:6.1f} dBFS → gate_ui = {gate_ui:.0f}")
    
    print("\n提示:")
    print("  - gate_ui 越小，C2 (高频增强) 越多")
    print("  - gate_ui 越大，C1 (低频增强) 越多")
    print("  - 建议从 p50 对应的值开始测试")
    print()

if __name__ == "__main__":
    main()
