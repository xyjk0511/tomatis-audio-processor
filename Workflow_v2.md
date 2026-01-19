# Tomatis D MNF 复刻流程 (v2 参数校准版)

该流程旨在将 `D MNF.flac` 处理成与物理录音 `Tomatis_D.flac` 动态一致的 `D_MNF_matched_v2.flac`。

## 1. 准备工作
确保以下文件在 `F:\TOMATIS\`：
- 原始文件：`D MNF.flac`
- 录音基准：`Tomatis_D.flac`
- 脚本：
  - `declick_inpaint.py`
  - `calibrate_to_baseline_v2.py`
  - `process_tomatis.py`

---

## 2. 制作“干净基准”

### 2.1 裁剪 (30分钟整)
从正式音乐起点 (16.80s) 截取 1800 秒。
```powershell
ffmpeg -y -ss 16.80 -i "Tomatis_D.flac" -t 1800 -ar 48000 -ac 2 -c:a flac -compression_level 8 "Tomatis_D_30m.flac"
```

### 2.2 去爆点 (De-click)
消除录音中的物理“啪啪”声，防止干扰校准。
```powershell
python declick_inpaint.py `
  -i "Tomatis_D_30m.flac" `
  -o "Tomatis_D_30m_declick.flac" `
  --k 14 --pad_ms 1.5 --merge_gap_ms 0.5 --max_fix_ms 8
```

---

## 3. 自动校准 (v2)

运行 v2 脚本，分析动态差异，计算正确的 `gate_offset` 和 `gain_db`。
```powershell
python calibrate_to_baseline_v2.py `
  --orig "D MNF.flac" `
  --base "Tomatis_D_30m_declick.flac" `
  --max_minutes 6 `
  --music_dbfs -65 `
  --out_json "calibration_v2.json"
```
**输出示例** (你的当前结果)：
- `gain_db`: -17.77 dB (录音偏轻)
- `gate_offset`: **-61.08** (实际生效阈值)
- `hyst_db`: 1.0

---

## 4. 最终处理 (Processing)

使用校准得到的参数应用到全长音频。
```powershell
# 注意：gate_offset, hyst_db 填入上一步 calibration_v2.json 中的数值
python process_tomatis.py `
  -i "D MNF.flac" `
  -o "D_MNF_matched_v2.flac" `
  --gate_ui 50 `
  --gate_scale 1 `
  --gate_offset -61.08 `
  --hyst_db 1.0 `
  --up_delay_ms 0 `
  --fc 1000 --slope 12 `
  --c1_low 5 --c1_high -5 `
  --c2_low -5 --c2_high 5 `
  --state_csv "D_MNF_matched_v2_state.csv"
```

## 5. 产出物
- **`D_MNF_matched_v2.flac`**: 最终成品。动态行为 (Gate) 与去噪后的基准录音高度一致。
