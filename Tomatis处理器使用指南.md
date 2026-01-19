# Tomatis 音频处理器使用指南

## 📋 功能说明

这个处理器实现了 Tomatis 效果的核心功能：

### C1 / C2 滤波器

- **C1** (安静段落): 低频增强 (+5dB)、高频衰减 (-5dB)
- **C2** (响亮段落): 低频衰减 (-5dB)、高频增强 (+5dB)

### Gate 门控

- 基于 RMS dBFS 自动切换 C1/C2
- 带回差（hysteresis）避免抖动
- C1→C2 有上行延迟（默认 250ms）
- C2→C1 立即切换

### 技术实现

- 短时 FFT (4096 点) + 频域增益
- Overlap-Add (OLA) 无缝拼接
- 支持双声道 48kHz FLAC

---

## 🚀 快速开始

### 基本用法

```powershell
conda activate dsp

python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_processed.flac"
```

### 带参数的完整示例

```powershell
python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_tomatis.flac" `
  --gate_ui 50 `
  --fc 1000 `
  --slope 12 `
  --c1_low 5 --c1_high -5 `
  --c2_low -5 --c2_high 5 `
  --up_delay_ms 250 `
  --hyst_db 3 `
  --state_csv "D_MNF_switch.csv"
```

---

## ⚙️ 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-i, --input` | 输入 FLAC 文件 | `"D MNF.flac"` |
| `-o, --output` | 输出 FLAC 文件 | `"output.flac"` |

### Gate 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gate_ui` | 50 | Gate UI 值 (0-100) |
| `--gate_offset` | -100 | Gate 偏移量（gate_ui=50 → -50 dBFS） |
| `--hyst_db` | 3.0 | 回差（dB），避免抖动 |
| `--up_delay_ms` | 250.0 | C1→C2 上行延迟（ms） |

### 滤波器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fc` | 1000.0 | 中心频率（Hz） |
| `--slope` | 12.0 | 坡度（dB/octave），可选 6/12/18 |
| `--c1_low` | 5.0 | C1 低频增益（dB） |
| `--c1_high` | -5.0 | C1 高频增益（dB） |
| `--c2_low` | -5.0 | C2 低频增益（dB） |
| `--c2_high` | 5.0 | C2 高频增益（dB） |

### FFT 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_fft` | 4096 | FFT 窗长（更大=更平滑但更慢） |
| `--hop` | 2048 | 跳步长度（通常是 n_fft 的一半） |

### 可选输出

| 参数 | 说明 |
|------|------|
| `--state_csv` | 输出状态 CSV 文件（记录每帧的 C1/C2 状态） |

---

## 📊 参数调整指南

### 调整 Gate 阈值

如果 C1/C2 切换不符合预期：

```powershell
# 提高阈值（更多 C1，更少 C2）
--gate_ui 60

# 降低阈值（更少 C1，更多 C2）
--gate_ui 40
```

### 减少切换抖动

如果切换太频繁或"吵"：

```powershell
# 增大回差
--hyst_db 6

# 增大 FFT 窗长（更平滑）
--n_fft 8192 --hop 4096
```

### 调整滤波器强度

```powershell
# 更强的效果（±10dB）
--c1_low 10 --c1_high -10 --c2_low -10 --c2_high 10

# 更温和的效果（±3dB）
--c1_low 3 --c1_high -3 --c2_low -3 --c2_high 3
```

### 改变坡度

```powershell
# 更陡的坡度（18 dB/octave）
--slope 18

# 更缓的坡度（6 dB/octave）
--slope 6
```

---

## 🔍 验证结果

### 1. 检查 C1/C2 占比

处理完成后，查看输出：

```
统计信息:
  总帧数: 45000
  C1 帧数: 25000 (55.6%)
  C2 帧数: 20000 (44.4%)
```

**正常范围**: C1 和 C2 都应该有一定占比（不应该是 0% 或 100%）

### 2. 查看状态 CSV

如果使用了 `--state_csv`，可以分析切换行为：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D_MNF_switch.csv")

# 绘制状态切换图
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(df['time_sec'], df['level_dbfs'])
plt.ylabel('Level (dBFS)')
plt.grid(True)

plt.subplot(2, 1, 2)
state_num = df['state'].map({'C1': 1, 'C2': 2})
plt.plot(df['time_sec'], state_num)
plt.ylabel('State (1=C1, 2=C2)')
plt.xlabel('Time (s)')
plt.yticks([1, 2], ['C1', 'C2'])
plt.grid(True)

plt.tight_layout()
plt.savefig('tomatis_state_analysis.png', dpi=150)
plt.show()
```

### 3. 听感验证

- **大动态段落**: 应该更"亮"（高频增强，C2 更多）
- **安静段落**: 应该更"厚"（低频增强，C1 更多）

---

## 🛠️ 故障排除

### 问题 1: FLAC 写入失败

**症状**: 输出 WAV 而不是 FLAC

**解决方案**: 使用 ffmpeg 转换

```powershell
ffmpeg -y -i output.wav -c:a flac -compression_level 8 output.flac
```

### 问题 2: C1 或 C2 占比为 0%

**原因**: Gate 阈值设置不当

**解决方案**: 
1. 先运行 `analyze_dbfs.py` 查看音频的 dBFS 范围
2. 调整 `--gate_ui` 到合适的值

### 问题 3: 切换太频繁

**解决方案**:
```powershell
# 增大回差和上行延迟
--hyst_db 6 --up_delay_ms 500
```

### 问题 4: 处理速度慢

**解决方案**:
```powershell
# 减小 FFT 窗长（但会降低平滑度）
--n_fft 2048 --hop 1024
```

---

## 📈 高级用法

### 批量处理

```powershell
# 批量处理多个文件
Get-ChildItem *.flac | ForEach-Object {
    $outname = $_.BaseName + "_tomatis.flac"
    python process_tomatis.py -i $_.Name -o $outname --gate_ui 50
}
```

### 对比不同参数

```powershell
# 生成多个版本对比
python process_tomatis.py -i "D MNF.flac" -o "output_gate40.flac" --gate_ui 40
python process_tomatis.py -i "D MNF.flac" -o "output_gate50.flac" --gate_ui 50
python process_tomatis.py -i "D MNF.flac" -o "output_gate60.flac" --gate_ui 60
```

### 校准 Gate 映射

如果需要匹配设备的实际行为：

```powershell
# 例如：gate=50 应该对应 -40 dBFS（而不是默认的 -50 dBFS）
--gate_ui 50 --gate_offset -90
```

---

## 📚 相关文件

- [process_tomatis.py](file:///F:/TOMATIS/process_tomatis.py) - 主处理脚本
- [analyze_dbfs.py](file:///F:/TOMATIS/analyze_dbfs.py) - dBFS 分析（用于确定 gate 阈值）
- [常见问题和下一步.md](file:///F:/TOMATIS/常见问题和下一步.md) - 环境配置说明

---

## 🎯 典型工作流程

1. **分析原始音频**
   ```powershell
   python analyze_dbfs.py
   ```
   查看 dBFS 范围，确定合适的 gate 值

2. **处理音频**
   ```powershell
   python process_tomatis.py -i "D MNF.flac" -o "output.flac" --gate_ui 50 --state_csv "state.csv"
   ```

3. **验证结果**
   - 检查 C1/C2 占比
   - 查看状态 CSV
   - 听感测试

4. **调整参数**（如需要）
   - 根据验证结果调整 gate、回差等参数
   - 重新处理

---

## ✨ 提示

- **首次使用**: 建议保留默认参数，先看效果
- **Gate 调整**: 从 `analyze_dbfs.py` 的结果开始，选择音频动态范围中间的值
- **平滑度**: 如果需要更平滑的切换，增大 `n_fft` 和 `hyst_db`
- **性能**: 处理 3 分钟音频大约需要 10-30 秒（取决于 FFT 参数）
