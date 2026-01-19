# Tomatis 处理器 - 快速测试

## 🚀 快速测试命令

### 测试 1: 使用默认参数
```powershell
conda activate dsp

python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis_default.flac" `
  --state_csv "state_default.csv"
```

### 测试 2: 调整 gate 阈值
```powershell
# Gate = 40 (更多 C2，更"亮")
python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis_gate40.flac" `
  --gate_ui 40 `
  --state_csv "state_gate40.csv"

# Gate = 60 (更多 C1，更"厚")
python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis_gate60.flac" `
  --gate_ui 60 `
  --state_csv "state_gate60.csv"
```

### 测试 3: 更强的效果
```powershell
python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis_strong.flac" `
  --c1_low 10 --c1_high -10 `
  --c2_low -10 --c2_high 10 `
  --state_csv "state_strong.csv"
```

### 测试 4: 更平滑的切换
```powershell
python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis_smooth.flac" `
  --hyst_db 6 `
  --n_fft 8192 --hop 4096 `
  --state_csv "state_smooth.csv"
```

---

## 📊 验证结果

### 查看统计信息
处理完成后，查看输出的 C1/C2 占比：
- 正常范围：两者都应该有一定占比（不是 0% 或 100%）

### 分析状态 CSV
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取状态数据
df = pd.read_csv("state_default.csv")

# 统计 C1/C2 占比
c1_count = (df['state'] == 'C1').sum()
c2_count = (df['state'] == 'C2').sum()
total = len(df)

print(f"C1: {c1_count} 帧 ({c1_count/total*100:.1f}%)")
print(f"C2: {c2_count} 帧 ({c2_count/total*100:.1f}%)")

# 绘制状态切换图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# 上图：电平
ax1.plot(df['time_sec'], df['level_dbfs'], linewidth=0.5)
ax1.set_ylabel('Level (dBFS)')
ax1.grid(True, alpha=0.3)
ax1.set_title('音频电平和状态切换')

# 下图：状态
state_num = df['state'].map({'C1': 1, 'C2': 2})
ax2.plot(df['time_sec'], state_num, linewidth=0.5)
ax2.set_ylabel('State')
ax2.set_xlabel('Time (s)')
ax2.set_yticks([1, 2])
ax2.set_yticklabels(['C1 (厚)', 'C2 (亮)'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tomatis_state_analysis.png', dpi=150)
print("\n图表已保存: tomatis_state_analysis.png")
plt.show()
```

### 对比原始和处理后的音频
```python
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# 读取音频
x_orig, sr = sf.read("D MNF.flac", dtype='float32')
x_proc, _ = sf.read("D_MNF_tomatis_default.flac", dtype='float32')

# 转单声道
if x_orig.ndim == 2:
    x_orig = x_orig.mean(axis=1)
if x_proc.ndim == 2:
    x_proc = x_proc.mean(axis=1)

# 对齐长度
L = min(len(x_orig), len(x_proc))
x_orig = x_orig[:L]
x_proc = x_proc[:L]

# 计算频谱
from scipy import signal
f, Pxx_orig = signal.welch(x_orig, sr, nperseg=4096)
f, Pxx_proc = signal.welch(x_proc, sr, nperseg=4096)

# 绘制频谱对比
plt.figure(figsize=(12, 6))
plt.semilogx(f, 10*np.log10(Pxx_orig + 1e-12), label='原始', alpha=0.7)
plt.semilogx(f, 10*np.log10(Pxx_proc + 1e-12), label='处理后', alpha=0.7)
plt.xlabel('频率 (Hz)')
plt.ylabel('功率谱密度 (dB)')
plt.title('原始 vs 处理后频谱对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([20, 20000])
plt.savefig('spectrum_comparison.png', dpi=150)
print("频谱对比图已保存: spectrum_comparison.png")
plt.show()
```

---

## ✅ 检查清单

处理完成后，检查：

- [ ] 输出文件已生成
- [ ] C1 和 C2 占比都不是 0% 或 100%
- [ ] 状态 CSV 已生成
- [ ] 听感测试：大动态段落更"亮"，安静段落更"厚"
- [ ] 没有明显的切换噪音或抖动

---

## 🔧 常见调整

### C2 太少（几乎全是 C1）
→ 降低 gate: `--gate_ui 40`

### C1 太少（几乎全是 C2）
→ 提高 gate: `--gate_ui 60`

### 切换太频繁/抖动
→ 增大回差: `--hyst_db 6`

### 效果不明显
→ 增强增益: `--c1_low 10 --c1_high -10 --c2_low -10 --c2_high 10`

---

## 📁 生成的文件

- `D_MNF_tomatis_*.flac` - 处理后的音频
- `state_*.csv` - 状态记录
- `tomatis_state_analysis.png` - 状态分析图（需运行 Python 脚本）
- `spectrum_comparison.png` - 频谱对比图（需运行 Python 脚本）
