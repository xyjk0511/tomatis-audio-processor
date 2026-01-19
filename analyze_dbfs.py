import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.signal import correlate

EPS = 1e-12

def to_mono(x):
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def rms_dbfs(frame):
    r = np.sqrt(np.mean(frame**2) + EPS)
    return 20 * np.log10(r + EPS)

def frame_dbfs(x, sr, frame_ms=20, hop_ms=10):
    frame = int(sr * frame_ms / 1000)
    hop   = int(sr * hop_ms / 1000)
    vals = []
    ts = []
    for i in range(0, len(x) - frame + 1, hop):
        vals.append(rms_dbfs(x[i:i+frame]))
        ts.append(i / sr)
    return np.array(ts), np.array(vals)

def align_by_xcorr(x, y, max_lag_s=2.0, sr=48000):
    # 在 +/- max_lag_s 范围内找相关峰值
    max_lag = int(max_lag_s * sr)
    y_seg = y
    corr = correlate(y_seg, x, mode='full')
    lags = np.arange(-len(x)+1, len(y_seg))
    # 限制 lag 范围
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lags_m = lags[mask]
    corr_m = corr[mask]
    best_lag = lags_m[np.argmax(corr_m)]
    # best_lag > 0 表示 y 比 x 晚（y 需要往前切）
    return int(best_lag)

def normalize_rms(x, target_rms):
    r = np.sqrt(np.mean(x**2) + EPS)
    if r < EPS:
        return x
    return x * (target_rms / r)

# ====== 你要改的路径 ======
# 文件说明：
# - D MNF.flac: 原始音频，未经任何处理
# - Tomatis_D.flac: 录制的音频，前 16 秒是其他音乐，需要裁剪
# - matlab_D_15db_1000Hz_12db.flac: Matlab 处理后的输出

input_path  = "D MNF.flac"
tomatis_path = "Tomatis_D_cut.flac"
matlab_path  = "matlab_D_15db_1000Hz_12db.flac"

print("正在读取音频文件...")
print(f"  输入文件: {input_path}")
print(f"  Tomatis 输出: {tomatis_path}")
print(f"  Matlab 输出: {matlab_path}")
print()

# ====== 读入（统一 float32）======
xin, sr_in = sf.read(input_path, dtype="float32")
xt, sr_t   = sf.read(tomatis_path, dtype="float32")
xm, sr_m   = sf.read(matlab_path, dtype="float32")

print("✓ 音频文件读取成功")
print(f"  输入: {len(xin)} 采样点, {sr_in} Hz")
print(f"  Tomatis: {len(xt)} 采样点, {sr_t} Hz")
print(f"  Matlab: {len(xm)} 采样点, {sr_m} Hz")
print()

xin = to_mono(xin); xt = to_mono(xt); xm = to_mono(xm)

# ====== 统一采样率到 input 的 sr（建议 48000）======
target_sr = sr_in
if sr_t != target_sr:
    print(f"正在重采样 Tomatis 音频: {sr_t} Hz → {target_sr} Hz")
    xt = librosa.resample(xt, orig_sr=sr_t, target_sr=target_sr)
if sr_m != target_sr:
    print(f"正在重采样 Matlab 音频: {sr_m} Hz → {target_sr} Hz")
    xm = librosa.resample(xm, orig_sr=sr_m, target_sr=target_sr)

sr = target_sr
print(f"✓ 采样率统一为: {sr} Hz")
print()

# ====== Tomatis 已经裁剪过，无需再次裁剪 ======
xt2 = xt
print(f"✓ 使用已裁剪的 Tomatis 音频")
print(f"  长度: {len(xt2)} 采样点 ({len(xt2)/sr:.2f} 秒)")
print()

# ====== 用互相关对齐（先对齐 matlab 输出；Tomatis 也对齐）======
print("正在计算互相关对齐...")
lag_m = align_by_xcorr(xin, xm, max_lag_s=2.0, sr=sr)
lag_t = align_by_xcorr(xin, xt2, max_lag_s=2.0, sr=sr)

print(f"✓ 对齐完成")
print(f"  Matlab lag: {lag_m} 采样点 ({lag_m/sr:.3f} 秒)")
print(f"  Tomatis lag: {lag_t} 采样点 ({lag_t/sr:.3f} 秒)")
print()

def apply_lag(x, y, lag):
    # 返回对齐后的 (x_aligned, y_aligned)
    if lag > 0:
        y = y[lag:]
        x = x[:len(y)]
    elif lag < 0:
        x = x[-lag:]
        y = y[:len(x)]
    else:
        L = min(len(x), len(y))
        x, y = x[:L], y[:L]
    L = min(len(x), len(y))
    return x[:L], y[:L]

xin_m, xm_a = apply_lag(xin, xm, lag_m)
xin_t, xt_a = apply_lag(xin, xt2, lag_t)

# ====== RMS 归一化（避免整体增益差影响门限推断）======
base_rms = np.sqrt(np.mean(xin_m**2) + EPS)
xm_a = normalize_rms(xm_a, base_rms)
xt_a = normalize_rms(xt_a, base_rms)

# ====== 帧级 dBFS ======
print("正在计算帧级 dBFS...")
ts, db_in_m = frame_dbfs(xin_m, sr)
_,  db_out_m = frame_dbfs(xm_a,  sr)

ts2, db_in_t = frame_dbfs(xin_t, sr)
_,   db_out_t = frame_dbfs(xt_a,  sr)

print(f"✓ dBFS 计算完成")
print(f"  Matlab: {len(ts)} 帧")
print(f"  Tomatis: {len(ts2)} 帧")
print()

# ====== 导出 CSV ======
print("正在导出 CSV...")
df_m = pd.DataFrame({"t": ts, "in_dbfs": db_in_m, "matlab_dbfs": db_out_m})
df_t = pd.DataFrame({"t": ts2, "in_dbfs": db_in_t, "tomatis_dbfs": db_out_t})
df_m.to_csv("dbfs_matlab.csv", index=False)
df_t.to_csv("dbfs_tomatis.csv", index=False)
print("✓ CSV 已保存: dbfs_matlab.csv, dbfs_tomatis.csv")
print()

# ====== 画图 ======
print("正在生成图表...")
plt.figure()
plt.plot(ts, db_in_m, label="input")
plt.plot(ts, db_out_m, label="matlab_out")
plt.xlabel("time (s)")
plt.ylabel("RMS dBFS")
plt.title("Input vs Matlab Output (aligned, RMS-normalized)")
plt.legend()
plt.tight_layout()
plt.savefig("dbfs_matlab.png", dpi=150)

plt.figure()
plt.plot(ts2, db_in_t, label="input")
plt.plot(ts2, db_out_t, label="tomatis_out")
plt.xlabel("time (s)")
plt.ylabel("RMS dBFS")
plt.title("Input vs Tomatis Output (aligned, RMS-normalized, cut 16s)")
plt.legend()
plt.tight_layout()
plt.savefig("dbfs_tomatis.png", dpi=150)

print("✓ 图表已保存: dbfs_matlab.png, dbfs_tomatis.png")
print()
print("=" * 60)
print("分析完成！")
print("=" * 60)
print("\n生成的文件:")
print("  1. dbfs_matlab.csv - Matlab 输出数据")
print("  2. dbfs_tomatis.csv - Tomatis 输出数据")
print("  3. dbfs_matlab.png - Matlab 对比图")
print("  4. dbfs_tomatis.png - Tomatis 对比图")
print("\n下一步:")
print("  运行 'python analyze_gate_threshold.py' 分析 gate 阈值")
print()
