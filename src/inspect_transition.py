import numpy as np
import soundfile as sf
import sys

def inspect_transition(file_path):
    print(f"Inspecting transition in {file_path} (10s - 20s)...")
    x, sr = sf.read(file_path, start=10*48000, stop=20*48000, dtype='float32')
    
    # Mono power average
    if x.ndim > 1:
        x_mono = np.sqrt(np.mean(x**2, axis=1))
    else:
        x_mono = x
        
    chunk_size = int(0.1 * sr) # 100ms chunks
    num_chunks = len(x_mono) // chunk_size
    
    print("\nTime (s) | Level (dBFS) | Status")
    print("-" * 40)
    
    for i in range(num_chunks):
        chunk = x_mono[i*chunk_size : (i+1)*chunk_size]
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        
        t = 10.0 + i * 0.1
        status = "SILENCE" if db < -60 else ("LOW" if db < -40 else "ACTIVE")
        bar = "#" * int((db + 100) / 4) if db > -80 else ""
        
        print(f"{t:6.1f}   | {db:6.1f}       | {status} {bar}")

if __name__ == "__main__":
    inspect_transition("Tomatis_D.flac")
