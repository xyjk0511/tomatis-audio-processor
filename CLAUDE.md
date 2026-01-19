# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Tomatis 音频处理系统 - 复刻 Tomatis 物理设备的动态门控倾斜滤波器处理。

技术栈: Python 3.11, NumPy, SciPy, SoundFile, Librosa, FFmpeg
采样率: 48kHz, 格式: FLAC 24-bit

## 环境配置

```powershell
# 一键安装
.\quick_setup.ps1

# 或手动
conda create -n dsp python=3.11 -y
conda activate dsp
pip install numpy scipy soundfile librosa pandas matplotlib

# 验证
python test_environment.py
```

## 核心命令

```powershell
# 激活环境
conda activate dsp

# 主处理器
python process_tomatis.py -i input.flac -o output.flac --gate_ui 50 --gate_offset -61.08 --hyst_db 1.0

# 自动校准
python calibrate_to_baseline_v2.py --orig source.flac --base baseline.flac --out_json calibration.json

# 去爆点
python declick_inpaint.py -i input.flac -o output.flac --k 14 --pad_ms 1.5

# 音频裁剪
ffmpeg -y -ss 16.80 -i input.flac -t 1800 -ar 48000 -ac 2 -c:a flac output.flac
```

## 架构

### 信号处理流程

```
输入 → Padding(n_fft/2) → STFT(4096,2048) → RMS计算 → Gate状态机 → 倾斜滤波 → ISTFT+OLA → 裁剪 → 输出
```

### Gate 状态机

- C1(安静): 低频+5dB, 高频-5dB
- C2(响亮): 低频-5dB, 高频+5dB
- 阈值公式: `T_dBFS = gate_scale * gate_ui + gate_offset`
- 带回差(hyst_db)和上行延迟(up_delay_ms)

### 核心模块

- `process_tomatis.py` - 主处理器 v1.3, STFT+OLA 实现
- `calibrate_to_baseline_v2.py` - 互相关对齐 + K-means 聚类校准
- `declick_inpaint.py` - MAD 检测 + 线性插值修复

### 关键参数

```python
n_fft = 4096      # FFT 窗长
hop = 2048        # 跳步
fc = 1000         # 中心频率 Hz
slope = 12        # 坡度 dB/octave
gate_offset = -61.08  # v2 校准值
```

## 设计决策

1. 多声道 RMS: 能量平均 `sqrt(mean(L^2+R^2)/2)` 而非简单平均
2. 边界处理: 前后 Padding + 精确裁剪, 消除 OLA 掉底
3. 倾斜滤波: 分段爬坡支持任意正负增益组合
