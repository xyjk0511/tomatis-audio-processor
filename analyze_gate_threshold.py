"""
Gate 阈值分析脚本
用于从 dBFS CSV 数据中自动检测 gate 切换点并估计阈值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_gate_threshold(csv_path, output_name, diff_threshold=3.0):
    """
    分析 gate 阈值
    
    参数:
        csv_path: CSV 文件路径
        output_name: 输出名称（用于图表标题）
        diff_threshold: 输出变化阈值（dB），超过此值认为是 gate 切换
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 计算输出的变化率
    out_col = [c for c in df.columns if 'out' in c or 'matlab' in c or 'tomatis' in c][0]
    df['out_diff'] = df[out_col].diff().abs()
    
    # 找出切换点
    switch_points = df[df['out_diff'] > diff_threshold].copy()
    
    if len(switch_points) == 0:
        print(f"\n{output_name}: 未检测到明显的 gate 切换点")
        return None
    
    # 统计分析
    gate_threshold_mean = switch_points['in_dbfs'].mean()
    gate_threshold_median = switch_points['in_dbfs'].median()
    gate_threshold_std = switch_points['in_dbfs'].std()
    
    print(f"\n{'='*60}")
    print(f"{output_name} - Gate 切换点分析")
    print(f"{'='*60}")
    print(f"检测到 {len(switch_points)} 个切换点")
    print(f"\n切换时的输入 dBFS 统计:")
    print(f"  平均值: {gate_threshold_mean:.2f} dB")
    print(f"  中位数: {gate_threshold_median:.2f} dB")
    print(f"  标准差: {gate_threshold_std:.2f} dB")
    print(f"  范围: [{switch_points['in_dbfs'].min():.2f}, {switch_points['in_dbfs'].max():.2f}] dB")
    
    print(f"\n前 5 个切换点:")
    print(switch_points[['t', 'in_dbfs', out_col, 'out_diff']].head().to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 子图 1: dBFS 曲线 + 切换点标记
    plt.subplot(2, 1, 1)
    plt.plot(df['t'], df['in_dbfs'], label='Input dBFS', alpha=0.7)
    plt.plot(df['t'], df[out_col], label='Output dBFS', alpha=0.7)
    plt.scatter(switch_points['t'], switch_points['in_dbfs'], 
                color='red', s=50, zorder=5, label='Gate 切换点')
    plt.axhline(y=gate_threshold_median, color='green', linestyle='--', 
                label=f'估计阈值: {gate_threshold_median:.1f} dB')
    plt.xlabel('Time (s)')
    plt.ylabel('dBFS')
    plt.title(f'{output_name} - dBFS 曲线与 Gate 切换点')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 输出变化率
    plt.subplot(2, 1, 2)
    plt.plot(df['t'], df['out_diff'], label='输出变化率', color='orange')
    plt.axhline(y=diff_threshold, color='red', linestyle='--', 
                label=f'检测阈值: {diff_threshold} dB')
    plt.scatter(switch_points['t'], switch_points['out_diff'], 
                color='red', s=50, zorder=5)
    plt.xlabel('Time (s)')
    plt.ylabel('输出变化 (dB)')
    plt.title('输出变化率（用于检测 gate 切换）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'gate_analysis_{output_name}.png', dpi=150)
    print(f"\n已保存图表: gate_analysis_{output_name}.png")
    
    return {
        'mean': gate_threshold_mean,
        'median': gate_threshold_median,
        'std': gate_threshold_std,
        'n_switches': len(switch_points),
        'switch_points': switch_points
    }

if __name__ == "__main__":
    analyze_gate_threshold("dbfs_tomatis.csv", "gate_analysis_device")
    print("="*60)
    
    # 分析 Matlab 输出
    result_matlab = analyze_gate_threshold(
        'dbfs_matlab.csv', 
        'Matlab',
        diff_threshold=3.0
    )
    
    # 分析 Tomatis 输出
    result_tomatis = analyze_gate_threshold(
        'dbfs_tomatis.csv', 
        'Tomatis',
        diff_threshold=3.0
    )
    
    # 对比总结
    print(f"\n{'='*60}")
    print("总结对比")
    print(f"{'='*60}")
    
    if result_matlab:
        print(f"Matlab 估计阈值: {result_matlab['median']:.2f} dB (±{result_matlab['std']:.2f} dB)")
    
    if result_tomatis:
        print(f"Tomatis 估计阈值: {result_tomatis['median']:.2f} dB (±{result_tomatis['std']:.2f} dB)")
    
    print("\n分析完成！")
