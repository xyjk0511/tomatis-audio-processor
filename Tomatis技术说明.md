# Tomatis 处理器 - 技术说明和注意事项

## ⚠️ 重要技术说明

### 1. dBFS 定义和测量

**本程序中所有 dB 均指 dBFS（满幅 0 dBFS）**

- **满幅定义**: 浮点音频的 ±1.0 对应 0 dBFS
- **正常电平**: 音频电平为负值（例如 -60 到 -10 dBFS）
- **电平测量**: 每帧使用 RMS dBFS
  - 公式: `RMS_dBFS = 20 * log10(sqrt(mean(x²)))`
  - 声道处理: 左右声道平均后计算

### 2. Gate 门控映射

**默认映射公式**: `T_dBFS = gate_ui + gate_offset`

示例:
- `gate_ui=50`, `gate_offset=-100` → `T_dBFS=-50 dBFS`
- `gate_ui=60`, `gate_offset=-100` → `T_dBFS=-40 dBFS`

**校准方法**:
如果实际设备的 gate=50 对应 -40 dBFS（而不是 -50 dBFS），则:
```powershell
--gate_ui 50 --gate_offset -90
```

### 3. 倾斜增益曲线原理

**核心公式**:
```
x = log2(f / fc)  # 距离中心频率的倍频程距离
```

- **中心频率 fc**: 增益为 0 dB
- **低频侧** (f < fc, x < 0): 增益朝 `low_gain_db` 走
- **高频侧** (f > fc, x > 0): 增益朝 `high_gain_db` 走
- **坡度 slope**: 每 octave 的增益变化量

**平台开始频率**:
- 高频平台: `f_hi = fc * 2^(|G_hi| / slope)`
- 低频平台: `f_lo = fc * 2^(-|G_lo| / slope)`

**示例** (fc=1000Hz, slope=12dB/oct, G_hi=+5dB):
```
f_hi = 1000 * 2^(5/12) ≈ 1335 Hz
```
即从 1335 Hz 开始，增益保持在 +5 dB 平台。

### 4. 回差（Hysteresis）机制

**目的**: 避免在阈值附近频繁切换

**实现**:
```
Ton  = T + hyst_db / 2  # 上行阈值 (C1→C2)
Toff = T - hyst_db / 2  # 下行阈值 (C2→C1)
```

**状态机逻辑**:
- 在 C1: 只有 `level >= Ton` 才触发切换到 C2
- 在 C2: 只有 `level <= Toff` 才切换回 C1

**示例** (T=-50dBFS, hyst_db=3dB):
```
Ton  = -50 + 1.5 = -48.5 dBFS
Toff = -50 - 1.5 = -51.5 dBFS
```

### 5. 上行延迟（Up Delay）

**目的**: C1→C2 延迟切换，C2→C1 立即切换

**实现**:
```python
# 触发时记录计划切换的样点位置
pending_c2_at = current_sample + up_delay_samples

# 到达时才切换
if current_sample >= pending_c2_at:
    state = C2
```

**优点**: 
- 按样点时间实现，与采样率解耦
- 比"等几帧"更精确

### 6. OLA (Overlap-Add) 归一化

**关键**: 保证输出幅度不变

**公式**:
```
y[n] = Σ y_k[n] / (Σ w²[n] + ε)
```

其中:
- `y_k[n]`: 第 k 帧在位置 n 的贡献
- `w²[n]`: 窗函数平方的累积
- `ε`: 小常数，避免除零

**实现要点**:
- 分析窗 = 合成窗 (Hann/Hanning)
- 维护 `w_buf` 累积 `win²`
- 输出时除以 `w_buf`

---

## 🔧 参数调整指南

### 选择 Gate 阈值

**步骤 1**: 分析音频 dBFS 分布
```powershell
python analyze_dbfs_simple.py -i "D MNF.flac"
```

**步骤 2**: 根据百分位数选择
- **p50 (中位数)**: 50% 的帧会触发 C2
- **p30**: 30% 的帧会触发 C2（更多 C1）
- **p70**: 70% 的帧会触发 C2（更多 C2）

**步骤 3**: 转换为 gate_ui
```
gate_ui = T_dBFS - gate_offset
```

示例: 如果 p50=-45 dBFS，gate_offset=-100:
```
gate_ui = -45 - (-100) = 55
```

### 调整回差

**症状**: 切换太频繁，听起来"抖动"

**解决方案**:
```powershell
# 增大回差到 6 dB
--hyst_db 6
```

### 调整平滑度

**症状**: 切换时有明显的"咔嗒"声

**解决方案**:
```powershell
# 增大 FFT 窗长（更平滑但更慢）
--n_fft 8192 --hop 4096
```

### 调整效果强度

**更强的效果**:
```powershell
--c1_low 10 --c1_high -10 --c2_low -10 --c2_high 10
```

**更温和的效果**:
```powershell
--c1_low 3 --c1_high -3 --c2_low -3 --c2_high 3
```

### 改变坡度

**更陡的坡度** (18 dB/oct):
- 平台更早出现
- 中频段变化更快
```powershell
--slope 18
```

**更缓的坡度** (6 dB/oct):
- 平台更晚出现
- 中频段变化更平滑
```powershell
--slope 6
```

---

## 📊 验证结果的标准

### 1. C1/C2 占比检查

**正常范围**: 两者都应该有一定占比

```
✓ 正常:
  C1: 25000 帧 (55.6%)
  C2: 20000 帧 (44.4%)

✗ 异常:
  C1: 45000 帧 (100.0%)  ← gate 阈值太高
  C2: 0 帧 (0.0%)

✗ 异常:
  C1: 0 帧 (0.0%)        ← gate 阈值太低
  C2: 45000 帧 (100.0%)
```

### 2. 状态切换检查

查看 `state_csv`:
- **正常**: 切换次数适中（几十到几百次）
- **异常**: 每帧都切换（回差太小）
- **异常**: 从不切换（阈值设置错误）

### 3. 听感验证

- **大动态段落**: 应该更"亮"（高频增强）
- **安静段落**: 应该更"厚"（低频增强）
- **切换处**: 不应该有明显的"咔嗒"声

---

## 🐛 已知问题和解决方案

### 问题 1: FLAC 写入失败

**症状**: 输出 WAV 而不是 FLAC

**原因**: Windows 上 libsoundfile 写 PCM_24 FLAC 可能失败

**解决方案**: 使用 ffmpeg 转换
```powershell
ffmpeg -y -i output.wav -c:a flac -compression_level 8 output.flac
```

### 问题 2: 处理速度估算

**影响因素**:
- `n_fft` 越大越慢
- CPU 单核/多核差异很大
- 音频长度

**参考速度** (单核，n_fft=4096):
- 3 分钟音频: 约 10-30 秒
- 30 分钟音频: 约 2-5 分钟

### 问题 3: 内存使用

**估算**:
- 30 分钟，48kHz，双声道: 约 350 MB 内存
- 主要用于 OLA 缓冲区

---

## 📈 性能优化建议

### 减小 FFT 窗长
```powershell
--n_fft 2048 --hop 1024
```
- 速度提升约 2 倍
- 但平滑度降低

### 流式处理
当前实现已经是流式的，每 10 秒读取一块，内存占用稳定。

---

## 🔬 高级用法

### 校准 Gate 映射

如果需要匹配实际设备:

1. 在设备上测试，记录 gate=50 时的实际切换行为
2. 用 `analyze_dbfs.py` 分析同一音频
3. 找到对应的 dBFS 值（例如 -40 dBFS）
4. 计算 gate_offset:
   ```
   gate_offset = T_dBFS - gate_ui = -40 - 50 = -90
   ```
5. 使用新的 offset:
   ```powershell
   --gate_ui 50 --gate_offset -90
   ```

### 自定义增益曲线

修改 `build_tilt_gain_db()` 函数可以实现:
- 非对称坡度（低频和高频用不同的 slope）
- 多段平台（例如低频 +10dB，中频 0dB，高频 +5dB）
- 任意形状的频响曲线

---

## 📚 相关文件

- [process_tomatis.py](file:///F:/TOMATIS/process_tomatis.py) - 主处理脚本
- [analyze_dbfs_simple.py](file:///F:/TOMATIS/analyze_dbfs_simple.py) - 快速 dBFS 分析
- [Tomatis处理器使用指南.md](file:///F:/TOMATIS/Tomatis处理器使用指南.md) - 使用文档
- [Tomatis快速测试.md](file:///F:/TOMATIS/Tomatis快速测试.md) - 测试场景
