# 解决 Conda ToS 问题并创建 dsp 环境

## 问题说明

Conda 需要接受服务条款（Terms of Service）才能创建新环境。

## 解决方案

### 步骤 1: 接受 Conda 服务条款

在 PowerShell 中运行以下命令（这会打开一个交互式界面）：

```powershell
conda create -n test_env python=3.11
```

当提示接受条款时，输入 `yes` 并按回车。然后可以删除这个测试环境：

```powershell
conda env remove -n test_env
```

### 步骤 2: 创建 dsp 环境

接受条款后，运行：

```powershell
conda create -n dsp python=3.11 -y
```

### 步骤 3: 激活环境并安装包

```powershell
conda activate dsp
pip install numpy scipy soundfile librosa pandas matplotlib
```

### 步骤 4: 验证安装

```powershell
python -c "import numpy, scipy, soundfile, librosa, pandas, matplotlib; print('所有包安装成功!')"
```

---

## 一键复制命令（按顺序执行）

```powershell
# 1. 创建 dsp 环境
conda create -n dsp python=3.11 -y

# 2. 激活环境
conda activate dsp

# 3. 安装所有包
pip install numpy scipy soundfile librosa pandas matplotlib

# 4. 验证
python -c "import numpy, scipy, soundfile, librosa, pandas, matplotlib; print('所有包安装成功!')"

# 5. 查看已安装的包
pip list
```

---

## 如果仍然遇到 ToS 问题

尝试以下方法之一：

### 方法 1: 使用 Anaconda Prompt

1. 在开始菜单搜索 "Anaconda Prompt" 或 "Anaconda PowerShell Prompt"
2. 打开后运行上面的命令

### 方法 2: 手动接受条款

运行任意一个 conda 命令，在提示时输入 `yes`:

```powershell
conda info
```

然后再创建 dsp 环境。

### 方法 3: 配置文件方法

创建或编辑 `.condarc` 文件：

```powershell
# 查看 conda 配置文件位置
conda config --show-sources

# 添加配置（如果上述方法都不行）
conda config --set auto_activate_base false
```

---

## 验证环境创建成功

```powershell
# 查看所有环境
conda env list

# 应该看到 dsp 环境列在其中
```

---

## Python 解释器路径

创建成功后，Python 解释器路径为：

```
C:\Users\55093\miniconda3\envs\dsp\python.exe
```

或者（如果使用 Anaconda）：

```
C:\Users\55093\anaconda3\envs\dsp\python.exe
```

在 Antigravity IDE 中选择这个路径作为 Python 解释器。
