# fingerprint.py
import numpy as np
import os
import librosa
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict

def get_peaks(y, sr, nperseg=2048, neighborhood_size=20, threshold_db=-40):
    f, t, Sxx = spectrogram(y, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    local_max = maximum_filter(Sxx_db, size=neighborhood_size)
    peaks = (Sxx_db == local_max) & (Sxx_db > threshold_db)
    peak_freq_idx, peak_time_idx = np.where(peaks)
    peaks_list = sorted(zip(peak_time_idx, peak_freq_idx))
    return peaks_list, f, t

def generate_hashes(peaks_list, f, t, fan_out=5, time_window=50):
    hashes = []
    for i, (t1_idx, f1_idx) in enumerate(peaks_list):
        for j in range(1, fan_out + 1):
            if i + j >= len(peaks_list):
                break
            t2_idx, f2_idx = peaks_list[i + j]
            dt = t2_idx - t1_idx
            if dt <= 0 or dt > time_window:
                continue
            f1 = f[f1_idx]
            f2 = f[f2_idx]
            hash_key = (round(f1, 1), round(f2, 1), int(dt))
            hashes.append((hash_key, t1_idx))
    return hashes

def build_database(song_files, sr=22050):
    database = {}
    for song_path in song_files:
        song_name = os.path.basename(song_path).replace('.mp3', '').replace('.wav', '')
        print(f"Indexing: {song_name}...")
        y, _ = librosa.load(song_path, sr=sr, mono=True)
        peaks_list, f, t = get_peaks(y, sr)
        hashes = generate_hashes(peaks_list, f, t)
        for hash_key, t_idx in hashes:
            if hash_key not in database:
                database[hash_key] = []
            database[hash_key].append((song_name, t_idx))
        print(f"  → {len(hashes)} hashes generated")
    return database

def match_from_array(y, sr, database):
    peaks_list, f, t = get_peaks(y, sr)
    query_hashes = generate_hashes(peaks_list, f, t)
    offset_counts = defaultdict(lambda: defaultdict(int))
    for hash_key, q_time in query_hashes:
        if hash_key in database:
            for song_name, db_time in database[hash_key]:
                offset = db_time - q_time
                offset_counts[song_name][offset] += 1
    scores = {}
    for song_name, offsets in offset_counts.items():
        scores[song_name] = max(offsets.values())
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0] if ranked else "No match", ranked, offset_counts