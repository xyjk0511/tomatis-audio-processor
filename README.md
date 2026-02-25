# Tomatis Audio Processor

Tomatis 音频处理器 - 动态门控倾斜滤波器的 Python 实现

## 功能特性

- **动态门控**：基于音频电平自动切换 C1/C2 滤波器
- **倾斜滤波器**：频率依赖的增益调整
- **两种模式**：
  - 标准版：复刻物理设备行为
  - 自适应版：自动优化阈值，平滑过渡

## 快速开始

### 环境要求

- Python 3.11+
- NumPy, SciPy, SoundFile

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install numpy scipy soundfile librosa pandas matplotlib
```

### 基本用法

**标准版**（复刻设备）：
```bash
python src/process_tomatis.py -i input.flac -o output.flac --gate_ui 50
```

**自适应版**（音质优化）：
```bash
python src/process_tomatis_adaptive.py -i input.flac -o output.flac
```

## 核心概念

### C1 / C2 滤波器

- **C1**（安静段落）：低频增强 +15dB，高频衰减 -15dB
- **C2**（响亮段落）：低频衰减 -15dB，高频增强 +15dB

### 门控机制

- 基于 RMS dBFS 自动切换状态
- 带回差（hysteresis）避免抖动
- 可配置上行延迟和最短保持时间

## 参数说明

### 标准版主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gate_ui` | 50 | 门控阈值 (0-100) |
| `--gate_mode` | log_percent | 门控模式（log_percent/linear） |
| `--hyst_db` | 3.0 | 回差（dB） |
| `--up_delay_ms` | 250.0 | C1→C2 延迟（ms） |
| `--fc` | 1000.0 | 中心频率（Hz） |
| `--slope` | 12.0 | 坡度（dB/octave） |

### 自适应版主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target_c2` | 0.5 | 目标 C2 占比 |
| `--min_hold_ms` | 250.0 | 最短保持时间（ms） |
| `--xfade_ms` | 500.0 | 交叉淡化时间（ms） |
| `--headroom_margin` | 2.0 | 预衰减余量（dB） |

## 技术实现

- **STFT + OLA**：短时傅里叶变换 + 重叠相加
- **频域处理**：在频域应用倾斜增益曲线
- **边界处理**：Padding 消除开头/结尾掉底
- **峰值保护**：自动限幅避免削波

## 文档

- [使用指南](docs/Tomatis处理器使用指南.md) - 详细参数说明和调整指南
- [技术说明](docs/Tomatis技术说明.md) - 算法原理和实现细节
- [工作日志](docs/TOMATIS_WORK_LOG.md) - 开发过程和关键发现

## 版本历史

### v1.4 (2026-01-20)
- 新增对数百分比门控模式
- 新增动态范围参数
- 优化门控阈值计算

### v1.3 (2026-01-18)
- 添加边界 padding 处理
- 添加输出增益补偿参数
- 修复开头/结尾掉底问题

### v2.0 (2026-01-24) - 自适应版
- 自适应阈值计算
- Crossfade 平滑过渡
- 预衰减 headroom 保护
- 最短保持时间控制

## 许可证

MIT License

## 作者

DSP 分析工具
