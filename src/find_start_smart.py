import numpy as np
import soundfile as sf
import librosa

def find_start_smart(file_path, search_duration=30.0):
    print(f"Loading first {search_duration}s of {file_path}...")
    y, sr = librosa.load(file_path, sr=None, duration=search_duration, mono=True)
    
    # Calculate RMS energy (frame size ~50ms)
    hop_length = int(sr * 0.05)
    frame_length = int(sr * 0.1)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calculate dBFS
    # Note: librosa loads as float32 normalized, but we want dB relative to full scale
    db = 20 * np.log10(rms + 1e-9)
    
    # Calculate Spectral Centroid (brightness)
    # Music usually has higher/more variable centroid than background noise
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    print("\nTime (s) | Level (dB) | Centroid (Hz) | Status")
    print("-" * 55)
    
    candidates = []
    
    # Dynamic thresholding
    # Assume first 1 second is noise if it's low level
    noise_floor = np.mean(db[:int(1.0/0.05)]) if len(db) > 20 else -80
    threshold = max(noise_floor + 15, -50) # At least -50dB or 15dB above floor
    
    print(f"Noise floor estimate: {noise_floor:.1f} dB")
    print(f"Trigger threshold: {threshold:.1f} dB")
    print("-" * 55)

    triggered = False
    start_time = 0.0
    
    # Calculate Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    for i, (t, val, c, z) in enumerate(zip(times, db, cent, zcr)):
        # Determine status
        if val < -60: status = "SILENCE"
        elif val < threshold: status = "NOISE"
        else: status = "MUSIC"
        
        # Simple visualization
        bar = "#" * int((val + 80)/2) if val > -80 else ""
        
        # FOCUS: Print everything between 14s and 19s
        if t >= 14.0 and t <= 19.0:
             print(f"{t:6.3f}   | {val:6.1f}     | {c:6.0f}        | {z:6.3f} | {status} {bar}")
             
             # Heuristic: Start is when centroid drops (music is bassier than hiss) AND level rises
             # OR when level jumps significantly
             
             # Check for level jump
             if i > 0 and (val - db[i-1]) > 5.0:
                 print(f"   >>> JUMP DETECTED (+{val - db[i-1]:.1f}dB) <<<")
                 candidates.append(t)
        
    return candidates

if __name__ == "__main__":
    candidates = find_start_smart("Tomatis_D.flac")
    
    if candidates:
        print(f"\nPotential Start Points: {[f'{c:.2f}s' for c in candidates]}")
    else:
        print("\nNo clear jump detected. Please review the table above.")
