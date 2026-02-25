# Tomatis 处理器 v1.3 - 快速测试（使用校准后的 gate 参数）

## 🎯 使用图中阈值的推荐参数

根据分析结果，阈值约为 -24.2 dBFS，使用以下参数：

```powershell
conda activate dsp

python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis_v13.flac" `
  --gate_ui 50 `
  --gate_scale 1.0 `
  --gate_offset -74.2 `
  --hyst_db 3 `
  --up_delay_ms 250 `
  --fc 1000 `
  --slope 12 `
  --c1_low 5 --c1_high -5 `
  --c2_low -5 --c2_high 5 `
  --n_fft 4096 --hop 2048 `
  --state_csv "state_v13.csv"
```

## ✅ v1.3 新特性验证清单

处理完成后，检查以下项目：

### 1. 输出长度验证
```powershell
# 检查输入和输出长度是否完全一致
python -c "import soundfile as sf; x_in, _ = sf.read('D MNF.flac'); x_out, _ = sf.read('D_MNF_tomatis_v13.flac'); print(f'输入: {len(x_in)} 采样点'); print(f'输出: {len(x_out)} 采样点'); print(f'一致: {len(x_in) == len(x_out)}')"
```

**预期**: 输出应该显示 "一致: True"

### 2. 边界 dBFS 检查
```python
import soundfile as sf
import numpy as np

def rms_dbfs(x):
    r = np.sqrt(np.mean(x**2) + 1e-12)
    return 20 * np.log10(r + 1e-12)

x, sr = sf.read("D_MNF_tomatis_v13.flac", dtype='float32')
if x.ndim == 2:
    x = np.sqrt(np.mean(x**2, axis=1))

# 检查开头和结尾
frame_size = 4096
head_dbfs = rms_dbfs(x[:frame_size])
tail_dbfs = rms_dbfs(x[-frame_size:])

print(f"开头 dBFS: {head_dbfs:.1f} dB")
print(f"结尾 dBFS: {tail_dbfs:.1f} dB")
print(f"\n✓ 正常" if head_dbfs > -100 and tail_dbfs > -100 else "✗ 异常掉底")
```

**预期**: 开头和结尾 dBFS 不应该掉到 -120 dBFS

### 3. 异常掉电平洞检查
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("state_v13.csv")

# 计算相邻帧的电平差
df['level_diff'] = df['level_dbfs'].diff().abs()

# 找出异常大的跳变
abnormal = df[df['level_diff'] > 20]

print(f"总帧数: {len(df)}")
print(f"异常跳变 (>20dB): {len(abnormal)} 个")

if len(abnormal) > 0:
    print("\n前 5 个异常点:")
    print(abnormal[['time_sec', 'level_dbfs', 'level_diff']].head())

# 绘制电平曲线
plt.figure(figsize=(14, 6))
plt.plot(df['time_sec'], df['level_dbfs'], linewidth=0.5, alpha=0.7)
if len(abnormal) > 0:
    plt.scatter(abnormal['time_sec'], abnormal['level_dbfs'], 
                color='red', s=20, zorder=5, label='异常跳变')
plt.xlabel('Time (s)')
plt.ylabel('Level (dBFS)')
plt.title('电平曲线 - 异常点检测')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('level_check_v13.png', dpi=150)
print("\n图表已保存: level_check_v13.png")
plt.show()
```

**预期**: 异常跳变应该显著减少（理想情况下接近 0）

### 4. 峰值溢出检查
```python
import soundfile as sf
import numpy as np

x, sr = sf.read("D_MNF_tomatis_v13.flac", dtype='float32')
peak = np.max(np.abs(x))

print(f"峰值: {peak:.6f}")
print(f"峰值 dBFS: {20*np.log10(peak):.2f} dB")

if peak > 0.999:
    print("\n⚠ 警告: 峰值接近或超过满幅，可能被裁剪")
else:
    print("\n✓ 峰值正常")
```

**预期**: 峰值应该 < 0.999（如果超过，需要添加防爆音保护）

---

## 📊 与之前版本对比

### v1.2 vs v1.3 主要改进

| 项目 | v1.2 | v1.3 |
|------|------|------|
| 输出长度 | 可能不一致 | ✓ 严格等于输入 |
| 开头/结尾 | -120 dBFS 掉底 | ✓ 正常电平 |
| 异常掉电平洞 | 较多 | ✓ 显著减少 |
| Padding | 无 | ✓ n_fft/2 |
| 写出方式 | 直接写出 | ✓ 裁剪写出 |
| OLA 覆盖 | 可能不完整 | ✓ 留 n_fft 尾巴 |

---

## 🔬 Gate 参数校准说明

### 当前参数推导

从图中阈值 -24.2 dBFS 推导：

```
T_dBFS = gate_scale * gate_ui + gate_offset
-24.2 = 1.0 * 50 + gate_offset
gate_offset = -74.2
```

### 如何调整

**增加 C2 占比**（更多高频增强）:
```powershell
--gate_ui 40  # 降低阈值
```

**减少 C2 占比**（更多低频增强）:
```powershell
--gate_ui 60  # 提高阈值
```

**校准到设备**:
1. 在设备上测试多个 gate_ui 值
2. 用 analyze_dbfs.py 找到对应的 dBFS 阈值
3. 拟合线性关系：`T = a*gate_ui + b`
4. 使用: `--gate_scale a --gate_offset b`

---

## 📈 下一步分析

运行完成后：

1. **生成新的 dBFS 分析**:
   ```powershell
   python analyze_dbfs.py
   ```

2. **对比 C1/C2 切换行为**:
   ```python
   import pandas as pd
   
   df = pd.read_csv("state_v13.csv")
   c1_pct = (df['state'] == 'C1').sum() / len(df) * 100
   c2_pct = (df['state'] == 'C2').sum() / len(df) * 100
   
   print(f"C1: {c1_pct:.1f}%")
   print(f"C2: {c2_pct:.1f}%")
   ```

3. **听感测试**: 对比原始音频和处理后的音频

---

## ⚠️ 注意事项

### Padding 的影响

- 第一帧从 -pad 开始（负数样点位置）
- CSV 中只记录落在 [0, total_frames) 的帧
- 输出严格裁剪到原始长度

### 关于 Shelving vs Tilt

当前实现使用"倾斜增益曲线"（对数频率线性爬坡 + 平台钳位），这是工程上可用的近似。

论文中的实现使用真正的 shelving 滤波器（FIR 51-101 taps），如需更贴近论文，可以后续替换。

---

## 🎵 预期效果

处理后的音频应该：

- ✓ 长度与原始完全一致
- ✓ 开头/结尾无掉底
- ✓ 大动态段落更"亮"（C2 高频增强）
- ✓ 安静段落更"厚"（C1 低频增强）
- ✓ 切换平滑，无明显噪音
