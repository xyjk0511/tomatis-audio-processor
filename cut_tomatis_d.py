"""
裁剪 Tomatis_D 前 16 秒的脚本
用于去除录音软件问题导致的前置音乐
"""

import soundfile as sf
import sys

def cut_audio(input_path, output_path, cut_seconds=16.0):
    """
    裁剪音频文件的前 N 秒
    
    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        cut_seconds: 裁剪的秒数
    """
    print(f"正在读取: {input_path}")
    
    # 读取音频
    x, sr = sf.read(input_path, dtype='float32')
    
    print(f"采样率: {sr} Hz")
    print(f"声道数: {x.shape[1] if x.ndim == 2 else 1}")
    print(f"原始长度: {len(x)} 采样点 ({len(x)/sr:.2f} 秒)")
    
    # 裁剪
    cut_samples = int(cut_seconds * sr)
    x_cut = x[cut_samples:]
    
    print(f"\n裁剪 {cut_seconds} 秒 ({cut_samples} 采样点)")
    print(f"裁剪后长度: {len(x_cut)} 采样点 ({len(x_cut)/sr:.2f} 秒)")
    
    # 保存
    print(f"\n正在保存: {output_path}")
    sf.write(output_path, x_cut, sr, format='FLAC', subtype='PCM_24')
    
    print("✓ 完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认参数
        input_path = "Tomatis_D.flac"
        output_path = "Tomatis_D_cut16s.flac"
        cut_seconds = 16.0
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        output_path = input_path.replace(".flac", "_cut16s.flac")
        cut_seconds = 16.0
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        cut_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
    
    cut_audio(input_path, output_path, cut_seconds)
