# DSP 开发环境配置 - 完整步骤

## 📌 总览

本指南提供**最简配置**步骤，让你快速在 Antigravity IDE 中开始 DSP 音频处理开发。

---

## ✅ 步骤 1: 安装 Miniconda

### 1.1 下载 Miniconda

访问: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

或在 PowerShell 中下载:

```powershell
# 下载到临时目录
$url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$output = "$env:TEMP\Miniconda3-Installer.exe"
Invoke-WebRequest -Uri $url -OutFile $output
Start-Process $output
```

### 1.2 安装 Miniconda

1. 运行下载的安装程序
2. 选择 "Just Me"
3. 安装路径保持默认（`C:\Users\你的用户名\miniconda3`）
4. **重要**: 勾选 "Add Miniconda3 to my PATH environment variable"
5. 点击 Install

### 1.3 验证安装

**关闭并重新打开** PowerShell，然后运行:

```powershell
conda --version
```

应该显示类似: `conda 24.1.2`

---

## ✅ 步骤 2: 创建 Python 环境

在 PowerShell 中运行以下命令:

```powershell
# 创建名为 dsp 的 Python 3.11 环境
conda create -n dsp python=3.11 -y

# 激活环境
conda activate dsp

# 安装所有必需的包
pip install numpy scipy soundfile librosa pandas matplotlib

# 验证安装
python -c "import numpy, scipy, soundfile, librosa, pandas, matplotlib; print('所有包安装成功!')"
```

如果看到 "所有包安装成功!"，说明 Python 环境配置完成！

---

## ✅ 步骤 3: 安装 FFmpeg

### 3.1 下载 FFmpeg

访问: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

或在 PowerShell 中下载:

```powershell
# 下载 FFmpeg
$url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$output = "$env:TEMP\ffmpeg.zip"
Invoke-WebRequest -Uri $url -OutFile $output

# 解压到 C:\ffmpeg
Expand-Archive -Path $output -DestinationPath "$env:TEMP\ffmpeg_temp" -Force

# 移动文件
$extracted = Get-ChildItem "$env:TEMP\ffmpeg_temp" -Directory | Select-Object -First 1
New-Item -ItemType Directory -Path "C:\ffmpeg" -Force
Copy-Item -Path "$($extracted.FullName)\*" -Destination "C:\ffmpeg" -Recurse -Force

# 清理
Remove-Item "$env:TEMP\ffmpeg.zip" -Force
Remove-Item "$env:TEMP\ffmpeg_temp" -Recurse -Force

Write-Host "FFmpeg 已解压到 C:\ffmpeg"
```

### 3.2 添加到 PATH

**方法 1: 使用 PowerShell（推荐）**

```powershell
# 获取当前用户 PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# 添加 FFmpeg
$newPath = "$currentPath;C:\ffmpeg\bin"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

Write-Host "FFmpeg 已添加到 PATH，请重启终端"
```

**方法 2: 手动添加**

1. 右键 "此电脑" → "属性"
2. "高级系统设置" → "环境变量"
3. 在用户变量中找到 `Path`，点击"编辑"
4. 点击"新建"，添加: `C:\ffmpeg\bin`
5. 点击"确定"保存

### 3.3 验证安装

**重启 PowerShell**，然后运行:

```powershell
ffmpeg -version
```

应该显示 FFmpeg 版本信息。

---

## ✅ 步骤 4: 配置 Antigravity IDE

### 4.1 选择 Python 解释器

1. 在 Antigravity IDE 中打开项目 `F:\TOMATIS`
2. 打开命令面板（`Ctrl+Shift+P`）
3. 输入: `Python: Select Interpreter`
4. 选择或手动输入路径:
   ```
   C:\Users\55093\miniconda3\envs\dsp\python.exe
   ```

### 4.2 验证 IDE 配置

1. 在 IDE 中打开终端（`` Ctrl+` ``）
2. 运行:

```powershell
conda activate dsp
python test_environment.py
```

如果所有测试通过，说明配置成功！

---

## 🎯 快速命令参考

### 激活 Python 环境
```powershell
conda activate dsp
```

### 运行 Python 脚本
```powershell
python your_script.py
```

### 转换音频格式
```powershell
# 单个文件
ffmpeg -i input.flac -ar 48000 -ac 1 -c:a pcm_s16le output.wav

# 批量转换当前目录所有 .flac 文件
Get-ChildItem *.flac | ForEach-Object {
    ffmpeg -i $_.Name -ar 48000 -ac 1 -c:a pcm_s16le "$($_.BaseName).wav"
}
```

---

## 🔍 验证完整环境

运行测试脚本:

```powershell
cd F:\TOMATIS
conda activate dsp
python test_environment.py
```

应该看到:
```
✓ NumPy          x.x.x
✓ SciPy          x.x.x
✓ SoundFile      x.x.x
✓ Librosa        x.x.x
✓ Pandas         x.x.x
✓ Matplotlib     x.x.x
✓ FFmpeg 可用
✓ 音频处理测试通过
🎉 所有测试通过！环境配置成功！
```

---

## ❓ 常见问题

### Q: conda 命令找不到

**A**: 
1. 确保安装时勾选了 "Add to PATH"
2. 重启终端
3. 或使用 "Anaconda Prompt" 代替 PowerShell

### Q: pip install 很慢

**A**: 使用清华镜像源:
```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy soundfile librosa pandas matplotlib
```

### Q: FFmpeg 命令找不到

**A**:
1. 确保已添加 `C:\ffmpeg\bin` 到 PATH
2. **重启终端**（环境变量需要重启才生效）
3. 验证: `where.exe ffmpeg`

### Q: Antigravity IDE 找不到 Python 解释器

**A**:
1. 手动输入完整路径: `C:\Users\55093\miniconda3\envs\dsp\python.exe`
2. 检查环境是否创建: `conda env list`
3. 重启 IDE

---

## 📂 项目文件说明

- `test_environment.py` - 环境验证脚本
- `README_SETUP.md` - 快速参考指南
- `setup_guide.md` - 详细配置文档（在 artifacts 目录）

---

## ✨ 配置完成后

你现在可以:

1. ✅ 使用 Python 进行音频处理
2. ✅ 使用 FFmpeg 转换音频格式
3. ✅ 在 Antigravity IDE 中调试代码
4. ✅ 运行 DSP 算法和分析

**开始开发吧！** 🚀
