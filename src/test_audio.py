import librosa
import numpy as np
import matplotlib.pyplot as plt

# 生成测试信号（440Hz 正弦波，1秒）
sr = 48000
duration = 1.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# 计算频谱
stft = librosa.stft(signal)
magnitude = np.abs(stft)

# 显示信息
print(f"采样率: {sr} Hz")
print(f"信号长度: {len(signal)} 采样点")
print(f"STFT 形状: {stft.shape}")
print(f"频率范围: 0 - {sr/2} Hz")

print("\n✓ 音频处理功能正常！")
