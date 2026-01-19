import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve

EPS = 1e-12

def power_mono(x_lr):
    # x_lr: [N,2]
    p = 0.5 * (x_lr[:,0]**2 + x_lr[:,1]**2)
    return np.sqrt(p + EPS)

def stft_mag_avg(x, sr, n_fft=4096, hop=2048):
    win = np.hanning(n_fft).astype(np.float32)
    n = len(x)
    n_frames = 1 + (n - n_fft) // hop
    mags = []
    for i in range(n_frames):
        st = i * hop
        fr = x[st:st+n_fft] * win
        X = np.fft.rfft(fr)
        mag = np.abs(X).astype(np.float32)
        mags.append(mag)
    mags = np.stack(mags, axis=0)  # [T,F]
    return mags.mean(axis=0)  # [F]

def band_energy(mag, freqs, f1, f2):
    m = (freqs >= f1) & (freqs < f2)
    return float(np.mean(mag[m]**2) + EPS)

def find_delay_by_corr(base_mono, cand_mono, sr, ds_sr=2000):
    # downsample for speed
    up, down = ds_sr, sr
    b = resample_poly(base_mono - base_mono.mean(), up, down).astype(np.float32)
    c = resample_poly(cand_mono - cand_mono.mean(), up, down).astype(np.float32)

    # corr valid: cand against reversed base
    corr = fftconvolve(c, b[::-1], mode="full")
    k = int(np.argmax(corr))
    # full-mode shift: shift = k - (len(b)-1) in downsampled samples
    shift_ds = k - (len(b) - 1)
    delay_samples = int(round(shift_ds * (sr / ds_sr)))
    return delay_samples

def align_pair(base_lr, cand_lr, delay):
    # delay >0 means cand starts later; need drop cand head or base head accordingly
    if delay > 0:
        cand_lr = cand_lr[delay:]
    elif delay < 0:
        base_lr = base_lr[-delay:]
    n = min(len(base_lr), len(cand_lr))
    return base_lr[:n], cand_lr[:n]

def rms_db(x):
    return 20*np.log10(np.sqrt(np.mean(x*x) + EPS) + EPS)

def main(base_path, cand_path, sr=48000, n_fft=4096, hop=2048):
    b_lr, sr1 = sf.read(base_path, dtype="float32", always_2d=True)
    c_lr, sr2 = sf.read(cand_path, dtype="float32", always_2d=True)
    assert sr1 == sr2 == sr and b_lr.shape[1]==2 and c_lr.shape[1]==2

    b = power_mono(b_lr)
    c = power_mono(c_lr)

    delay = find_delay_by_corr(b, c, sr)
    print(f"[ALIGN] delay_samples (cand - base) = {delay} ({delay/sr*1000:.2f} ms)")

    b_lr2, c_lr2 = align_pair(b_lr, c_lr, delay)
    b2 = power_mono(b_lr2)
    c2 = power_mono(c_lr2)

    # ---- level align by 300-3kHz anchor (use avg spectrum)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    b_mag = stft_mag_avg(b2, sr, n_fft, hop)
    c_mag = stft_mag_avg(c2, sr, n_fft, hop)

    Eb = band_energy(b_mag, freqs, 300, 3000)
    Ec = band_energy(c_mag, freqs, 300, 3000)
    gain_lin = np.sqrt(Eb / Ec)
    gain_db = 20*np.log10(gain_lin + EPS)
    print(f"[LEVEL] anchor gain to apply on cand = {gain_db:.2f} dB (multiply by {gain_lin:.4f})")

    c_lr2_scaled = c_lr2 * gain_lin
    c2s = power_mono(c_lr2_scaled)

    # ---- spectrum diff (base - cand)
    c_mag2 = stft_mag_avg(c2s, sr, n_fft, hop)
    diff_db = 20*np.log10((b_mag+EPS)/(c_mag2+EPS))

    # summary bands
    bands = [(200,1000),(1000,3000),(3000,8000),(8000,16000)]
    for f1,f2 in bands:
        m = (freqs>=f1)&(freqs<f2)
        print(f"[BAND {f1}-{f2}Hz] mean ΔdB (base-cand) = {diff_db[m].mean():.2f} dB, std={diff_db[m].std():.2f}")

    # ---- residual
    res = b_lr2 - c_lr2_scaled
    res_m = power_mono(res)
    snr = rms_db(b2) - rms_db(res_m)
    print(f"[RESIDUAL] SNR (base vs residual) ≈ {snr:.2f} dB  (越大越像)")

    # save diff spectrum
    out = np.stack([freqs, diff_db], axis=1)
    np.savetxt("diff_spectrum.csv", out, delimiter=",", header="freq_hz,delta_db_base_minus_cand", comments="")
    print("[OUT] wrote diff_spectrum.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="Base audio file")
    parser.add_argument("cand", help="Candidate audio file")
    args = parser.parse_args()
    main(args.base, args.cand)
