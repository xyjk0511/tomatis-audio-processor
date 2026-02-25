"""
Inspect Tomatis_D.flac amplitude profile
Generates a plot and text report of the audio levels to identify music segments.
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import argparse

def rms_dbfs(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)

def inspect(file_path):
    print(f"Reading {file_path}...")
    x, sr = sf.read(file_path, dtype='float32')
    
    if x.ndim > 1:
        # Power average for RMS
        x_mono = np.sqrt(np.mean(x**2, axis=1))
    else:
        x_mono = x

    duration = len(x_mono) / sr
    print(f"Duration: {duration:.2f} s")
    print(f"Sample Rate: {sr} Hz")
    
    # Analyze in 0.5s chunks
    chunk_size = int(0.5 * sr)
    num_chunks = int(np.ceil(len(x_mono) / chunk_size))
    
    times = []
    levels = []
    
    print("\nAmplitude Profile (First 60s):")
    print("Time (s) | Level (dBFS) | Status")
    print("-" * 40)
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(x_mono))
        chunk = x_mono[start:end]
        
        lvl = rms_dbfs(chunk)
        t = i * 0.5
        
        times.append(t)
        levels.append(lvl)
        
        if t < 60:
            status = "SILENCE" if lvl < -60 else ("LOW" if lvl < -40 else "ACTIVE")
            bar = "#" * int((lvl + 100) / 5) if lvl > -100 else ""
            print(f"{t:6.1f}   | {lvl:6.1f}       | {status} {bar}")

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(times, levels)
    plt.axhline(y=-60, color='r', linestyle='--', label='Silence Threshold (-60dB)')
    plt.axhline(y=-40, color='orange', linestyle='--', label='Low Threshold (-40dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Level (dBFS)')
    plt.title(f'Amplitude Profile: {file_path}')
    plt.grid(True)
    plt.legend()
    
    output_png = "tomatis_d_profile.png"
    plt.savefig(output_png)
    print(f"\nPlot saved to {output_png}")

if __name__ == "__main__":
    inspect("Tomatis_D.flac")
