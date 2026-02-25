# Tomatis 项目工作日志

> 每次开新窗口请先阅读此文档，了解项目进展和关键发现。

## 项目目标

复刻 Tomatis 物理设备的动态门控倾斜滤波器处理，使软件处理结果与设备输出一致。

## 关键文件

| 文件 | 说明 |
|------|------|
| `D MNF.flac` | 原始输入音频 (30分钟) |
| `Tomatis_D.flac` | 设备原始输出 (有68.71秒偏移) |
| `Tomatis_D_30m_declick.flac` | 设备输出 (裁剪+去爆点) |
| `matlab_D_15db_1000Hz_12db.flac` | 东南大学团队Matlab实现 |
| `源文件.flac` | 用户的新测试源文件 |
| `process_tomatis.py` | 主处理器 (v1.4) |

---

## 设备参数 (用户提供)

```
音量：100
门控：50 (对数百分比，不是分贝！)
无延迟无提前
通道1：5，-5 (对应 C1: +15dB/-15dB)
通道2：-5，5 (对应 C2: -15dB/+15dB)
中心频率：1000Hz
坡度：12dB/oct
```

---

## 关键发现

### 1. 门控单位是"对数百分比" (2026-01-20)

**重要**: 门控值不是直接的dBFS，而是对数百分比！

- 分母是声音的最大强度 (0 dBFS)
- gate_ui 表示在对数刻度上的百分比位置

**公式**:
```
T_dBFS = -dynamic_range + dynamic_range * gate_ui / 100
```

**验证** (dynamic_range = 80dB):
- gate_ui = 0 → -80 dBFS (最敏感)
- gate_ui = 50 → -40 dBFS (中间) ✓ 实测验证
- gate_ui = 100 → 0 dBFS (最不敏感)

### 2. 反推的Gate阈值

通过分析设备输入输出，反推Gate切换点：
- C1→C2 切换: 约 -38.5 dBFS
- C2→C1 切换: 约 -41.5 dBFS
- **基础阈值: -40 dBFS**
- **回差: 约 3 dB**

### 3. 倾斜滤波器验证

| 参数 | 理论值 | 实测值 |
|------|--------|--------|
| C1 倾斜 | -30 dB | -18.9 dB |
| C2 倾斜 | +30 dB | +10.8 dB |
| C1-C2 差异 | -60 dB | -29.6 dB |

实测约为理论值的一半，可能是UI值"5"对应7.5dB而非15dB。
但使用±15dB处理后，与设备输出的频谱差异最小。

### 4. 与设备输出对比 (使用±15dB)

| 频段 | 差异 (设备-我们) |
|------|-----------------|
| 200-1000Hz | +0.07 dB ✓ |
| 1000-3000Hz | +0.00 dB ✓ |
| 3000-8000Hz | +1.15 dB |
| 8000-16000Hz | +3.54 dB |

低中频完美匹配，超高频略有差异。

---

## 代码更新记录

### v1.4 (2026-01-20)
- 新增 `gate_ui_to_dbfs_log_percent()` 函数
- 新增 `--gate_mode` 参数: `log_percent`(默认) / `linear`
- 新增 `--dynamic_range` 参数: 默认80dB
- 默认使用对数百分比模式

### v1.3 (2026-01-18)
- 添加 pad/trim 边界处理，消除开头/结尾掉底
- 添加 `--output_gain_db` 参数

---

## 当前最佳处理命令

```powershell
# 使用对数百分比模式 (推荐)
python process_tomatis.py -i input.flac -o output.flac --gate_ui 50

# 完整参数
python process_tomatis.py -i input.flac -o output.flac \
    --gate_ui 50 \
    --gate_mode log_percent \
    --dynamic_range 80 \
    --c1_low 15 --c1_high -15 \
    --c2_low -15 --c2_high 15 \
    --fc 1000 --slope 12 \
    --hyst_db 3 \
    --up_delay_ms 250
```

---

## 待解决问题

1. **超高频差异**: 8k-16kHz 偏高约3.5dB，可能需要高频滚降补偿
2. **上行延迟**: 用户说"无延迟无提前"，但我们设置250ms，需确认
3. **bone/air双通道**: 设备有骨传导和气传导两个输出，目前只处理单输出

---

## 参考资料

- Excel分析表格: `f:\xwechat_files\...\output.xlsx` (另一人的分析)
- 计划文件: `C:\Users\55093\.claude\plans\spicy-enchanting-eich.md`

---

*最后更新: 2026-01-20*
