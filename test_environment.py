#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境验证测试脚本
用于验证 DSP 开发环境是否正确配置
"""

import sys
import subprocess

def test_imports():
    """测试所有必需的 Python 包"""
    print("=== 测试 Python 包导入 ===\n")
    
    packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('soundfile', 'SoundFile'),
        ('librosa', 'Librosa'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_success = True
    
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:15} {version}")
        except ImportError as e:
            print(f"✗ {display_name:15} 导入失败: {e}")
            all_success = False
    
    return all_success

def test_ffmpeg():
    """测试 FFmpeg 是否可用"""
    print("\n=== 测试 FFmpeg ===\n")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # 提取版本信息（第一行）
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg 可用")
            print(f"  {version_line}")
            return True
        else:
            print(f"✗ FFmpeg 运行失败")
            return False
            
    except FileNotFoundError:
        print("✗ FFmpeg 未找到，请确保已安装并添加到 PATH")
        return False
    except Exception as e:
        print(f"✗ FFmpeg 测试失败: {e}")
        return False

def test_audio_processing():
    """测试基本音频处理功能"""
    print("\n=== 测试音频处理功能 ===\n")
    
    try:
        import numpy as np
        import librosa
        
        # 生成测试信号（1秒，440Hz 正弦波）
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        
        # 测试 librosa 功能
        stft = librosa.stft(signal)
        print(f"✓ 生成测试信号: {len(signal)} 采样点")
        print(f"✓ STFT 计算成功: {stft.shape}")
        
        # 测试频谱分析
        freqs = librosa.fft_frequencies(sr=sr)
        print(f"✓ 频率分析: {len(freqs)} 个频率点")
        
        return True
        
    except Exception as e:
        print(f"✗ 音频处理测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("DSP 开发环境验证")
    print("=" * 50)
    print()
    
    # Python 版本
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print()
    
    # 运行所有测试
    results = []
    results.append(("Python 包", test_imports()))
    results.append(("FFmpeg", test_ffmpeg()))
    results.append(("音频处理", test_audio_processing()))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:15} {status}")
    
    print()
    
    if all_passed:
        print("🎉 所有测试通过！环境配置成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
