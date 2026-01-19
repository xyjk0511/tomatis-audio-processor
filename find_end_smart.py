import numpy as np
import soundfile as sf
import librosa

def find_end_smart(file_path, search_duration=30.0):
    # Get total duration
    info = sf.info(file_path)
    total_dur = info.frames / info.samplerate
    start_search = max(0, total_dur - search_duration)
    
    print(f"File duration: {total_dur:.2f}s")
    print(f"Scanning end from {start_search:.2f}s to {total_dur:.2f}s...")
    
    y, sr = librosa.load(file_path, sr=None, offset=start_search, duration=search_duration, mono=True)
    
    # Calculate RMS energy
    hop_length = int(sr * 0.1) # 100ms
    frame_length = int(sr * 0.2)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = 20 * np.log10(rms + 1e-9)
    
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length) + start_search
    
    print("\nTime (s) | Level (dB) | Status")
    print("-" * 40)
    
    # Heuristic: Find the LAST point where signal is "MUSIC"
    # Noise floor from very end assumption
    if len(db) > 10:
        noise_floor = np.mean(db[-10:])
    else:
        noise_floor = -80
        
    threshold = max(noise_floor + 15, -60)
    print(f"Est. Tail Noise Floor: {noise_floor:.1f} dB")
    print(f"Silence Threshold: {threshold:.1f} dB")

    last_music_time = total_dur
    found_end = False
    
    # Iterate BACKWARDS
    for i in range(len(db)-1, -1, -1):
        t = times[i]
        val = db[i]
        
        status = "MUSIC" if val > threshold else "SILENCE"
        bar = "#" * int((val + 80)/2) if val > -80 else ""
        
        # Visualize last few seconds
        if t > total_dur - 10.0:
            print(f"{t:6.2f}   | {val:6.1f}     | {status} {bar}")
        
        if not found_end and val > threshold:
            # We found the last 'loud' moment
            # Check if it sustains for a bit (0.5s) to avoid clicks
            if i > 5 and np.mean(db[i-5:i]) > threshold:
                last_music_time = t
                found_end = True
                print(f"\n>>> AUDIO ENDS around {t:.2f}s (Level: {val:.1f}dB) <<<")

    return last_music_time, total_dur

if __name__ == "__main__":
    end_time, total = find_end_smart("Tomatis_D.flac")
    
    if end_time < total - 0.5:
        # Add 1.0s reverb tail
        cut_point = min(total, end_time + 1.0)
        print(f"\nRecommended End Cut: {cut_point:.2f} seconds")
        print(f"(Includes 1.0s tail/fade-out margin)")
    else:
        print("\nNo silence detected at end.")
