# DSP 开发环境配置说明

## 🚀 快速开始

### 方法一：自动安装（推荐）

在 PowerShell 中运行：

```powershell
cd F:\TOMATIS
.\scripts\quick_setup.ps1
```

这将自动安装 Miniconda、Python 环境和 FFmpeg。

### 方法二：分步安装

```powershell
# 1. 安装 Miniconda 和 Python 环境
.\setup_miniconda.ps1

# 2. 安装 FFmpeg
.\setup_ffmpeg.ps1

# 3. 验证环境
conda activate dsp
python test_environment.py
```

## 📝 安装后配置

### 1. 在 Antigravity IDE 中选择 Python 解释器

- 打开命令面板（`Ctrl+Shift+P`）
- 搜索 "Python: Select Interpreter"
- 选择: `C:\Users\55093\miniconda3\envs\dsp\python.exe`

### 2. 验证环境

```powershell
conda activate dsp
python test_environment.py
```

## 🔧 常用命令

### 激活环境
```powershell
conda activate dsp
```

### 音频格式转换
```powershell
ffmpeg -i input.flac -ar 48000 -ac 1 -c:a pcm_s16le output.wav
```

### 批量转换
```powershell
Get-ChildItem *.flac | ForEach-Object {
    ffmpeg -i $_.Name -ar 48000 -ac 1 -c:a pcm_s16le "$($_.BaseName).wav"
}
```

## 📚 详细文档

查看完整配置指南: [setup_guide.md](file:///C:/Users/55093/.gemini/antigravity/brain/7f6c129f-5ac1-4749-8482-0ff3180d1b6d/setup_guide.md)
