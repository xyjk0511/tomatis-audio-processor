
# Tomatis D MNF 复刻流程 (v2 参数校准版)

该流程旨在将 `D MNF.flac` 处理成与物理录音 `Tomatis_D.flac` 动态一致且音色匹配的 `D_MNF_matched_final.flac`。

## 1. 准备工作
确保以下文件在 `F:\TOMATIS\`：
- 原始文件：`D MNF.flac`
- 录音基准：`Tomatis_D.flac`
- 脚本：
  - `declick_inpaint.py`
  - `calibrate_to_baseline_v2.py`
  - `process_tomatis.py`
  - `layer2_analyze_eq.py`
  - `layer2b_apply_residual_eq_safe.py` (v2/SafeB 修正版)

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

## 3. 自动校准 (v2) - Layer 1

运行 v2 脚本，分析动态差异，计算正确的 `gate_offset` 和 `gain_db`。
```powershell
python calibrate_to_baseline_v2.py `
  --orig "D MNF.flac" `
  --base "Tomatis_D_30m_declick.flac" `
  --max_minutes 6 `
  --music_dbfs -65 `
  --out_json "calibration_v2.json"
```
**输出示例** (参考)：
- `gain_db`: -17.77 dB (录音偏轻)
- `gate_offset`: **-61.08** (实际生效阈值)
- `hyst_db`: 1.0

---

## 4. 基础处理 (Layer 1 Processing)

使用校准得到的参数应用到全长音频，生成动态匹配版本。
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
得到：`D_MNF_matched_v2.flac` (动态已对齐，但音色未匹配)

---

## 5. 频响分析 (Layer 2)

计算与基准的频谱差异，去除整体增益后的 EQ 曲线。
```powershell
python layer2_analyze_eq.py `
  --base "Tomatis_D_30m_declick.flac" `
  --target "D_MNF_matched_v2.flac" `
  --out_csv "layer2_eq_curve.csv" `
  --out_png "layer2_eq_curve.png"
```
得到：`layer2_eq_curve.csv`

---

## 6. EQ 补偿 - 初步应用

先应用基础 EQ（带峰值保护）。
```powershell
python layer2_apply_eq.py `
  -i "D_MNF_matched_v2.flac" `
  -o "D_MNF_matched_v2_eq_gp.flac" `
  --eq_csv "layer2_eq_curve.csv"
```
得到：`D_MNF_matched_v2_eq_gp.flac` (音色接近，但高频可能仍有偏差)

---

## 7. 精细修正 (Layer 2 Refinement Safe-B) - **推荐**

更安全的修正，强制 3kHz 以上不进行额外噪声补偿，只对齐中低频。
```powershell
# 需要 diff_spectrum.csv (由 compare_audio.py 生成)
python compare_audio.py "Tomatis_D_30m_declick.flac" "D_MNF_matched_v2_eq_gp.flac"

python layer2b_apply_residual_eq_safe.py `
  --in_audio "D_MNF_matched_v2_eq_gp.flac" `
  --out_audio "D_MNF_matched_v2_eq_gp_residual_safeB.flac" `
  --diff_csv "diff_spectrum.csv" `
  --clamp_hi 1.0 --hf_start 3000
```

---

## 8. 产出物选择

最终会有三个版本：
1.  **基础版**: `D_MNF_matched_v2_eq_gp.flac` (较暗)
2.  **均衡版**: `..._residual_eq_v2.flac` (3k-8k 略亮)
3.  **纯净版 (Safe B)**: `..._residual_safeB.flac` (推荐，绝对无底噪)

建议保留 Safe B 版本作为交付物。
