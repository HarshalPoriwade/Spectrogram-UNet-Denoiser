import tensorflow as tf
import soundfile as sf
import numpy as np
import scipy.signal
import glob
import os

GLOBAL_RIR_PATHS = []

# High Definition 48kHz
SAMPLE_RATE = 48000 
# 33536 samples without center padding yields EXACTLY 128 frames for STFT:
# 1 + (33536 - 1024) // 256 = 128.
SAMPLES_PER_FRAGMENT = 33536

def _load_raw_audio(file_path_str):
    """Loads audio from a file, converts to mono, and resamples to 48kHz."""
    data, sr = sf.read(file_path_str, dtype='float32')
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Convert multi-channel (e.g. DEMAND 16-ch) to Mono
    if sr != SAMPLE_RATE:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
    return data

def _slice_to_fragment(data):
    """Slices or pads audio to exactly SAMPLES_PER_FRAGMENT samples."""
    if len(data) > SAMPLES_PER_FRAGMENT:
        start = np.random.randint(0, len(data) - SAMPLES_PER_FRAGMENT)
        data = data[start : start + SAMPLES_PER_FRAGMENT]
    else:
        pad = SAMPLES_PER_FRAGMENT - len(data)
        data = np.pad(data, (0, pad), 'constant')
    return data.astype(np.float32)

def py_read_clean_audio(file_path):
    """Loads CLEAN audio and optionally applies Room Impulse Response (Echo simulation)."""
    file_path = file_path.decode('utf-8')
    try:
        data = _load_raw_audio(file_path)
        
        # Apply RIR to clean audio for dereverberation learning.
        if len(GLOBAL_RIR_PATHS) > 0 and np.random.rand() > 0.5:
            rir_path = np.random.choice(GLOBAL_RIR_PATHS)
            try:
                rir = _load_raw_audio(rir_path)
                rir = rir[:int(SAMPLE_RATE * 0.5)]  # Keep RIR to 0.5 sec max
                data_rev = scipy.signal.fftconvolve(data, rir, mode='full')
                data_rev = data_rev[:len(data)]  # Trim back to original length
                max_val = np.max(np.abs(data_rev))
                if max_val > 0:
                    data = data_rev / max_val  # Normalize after reverb
            except Exception:
                pass  # If RIR fails, just use dry (non-reverbed) audio

        return _slice_to_fragment(data)
    except Exception as e:
        print(f"[WARN] Error reading clean audio {file_path}: {e}")
        return np.zeros(SAMPLES_PER_FRAGMENT, dtype=np.float32)

def py_read_noise_audio(file_path):
    """Loads background noise audio."""
    file_path = file_path.decode('utf-8')
    try:
        data = _load_raw_audio(file_path)
        return _slice_to_fragment(data)
    except Exception as e:
        print(f"[WARN] Error reading noise audio {file_path}: {e}")
        return np.zeros(SAMPLES_PER_FRAGMENT, dtype=np.float32)

def mix_audio(clean, noise, target_snr_db):
    """Mixes clean voice and noise at a target SNR level."""
    clean_power = tf.reduce_mean(clean**2) + 1e-8
    noise_power = tf.reduce_mean(noise**2) + 1e-8
    
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    scale = tf.sqrt(clean_power / (noise_power * snr_linear))
    
    mixed = clean + noise * scale
    
    # Dynamic normalization to prevent IRM distortion at low SNRs.
    max_val = tf.reduce_max(tf.abs(mixed))
    scale_factor = tf.cond(max_val > 0.95, lambda: 0.95 / max_val, lambda: 1.0)
    
    mixed = mixed * scale_factor
    clean = clean * scale_factor
    
    # Return as 1D sequence for STFT (Batch, 33536)
    return mixed, clean

def create_tf_dataset(clean_dir, noise_dirs, batch_size=8, rir_dir=None):
    global GLOBAL_RIR_PATHS

    # Load RIR files (only real/simulated echoes, NOT pointsource noise)
    if rir_dir and os.path.exists(rir_dir):
        all_rirs = glob.glob(os.path.join(rir_dir, '**', '*.wav'), recursive=True)
        GLOBAL_RIR_PATHS = [p for p in all_rirs if 'pointsource' not in p.replace('\\', '/').lower()]
        print(f"Found {len(GLOBAL_RIR_PATHS)} RIR files for echo augmentation.")
    else:
        print("No RIR directory found. Skipping echo augmentation.")

    # Scan clean dataset (VCTK wav48 folder)
    print("Scanning clean directory...")
    clean_paths = (
        glob.glob(os.path.join(clean_dir, '**', '*.flac'), recursive=True) +
        glob.glob(os.path.join(clean_dir, '**', '*.wav'), recursive=True)
    )

    # Scan noise datasets (MUSAN music+noise only, UrbanSound8K, DEMAND 48k, ESC-50)
    print("Scanning noise directories...")
    noise_paths = []
    if isinstance(noise_dirs, list):
        for nd in noise_dirs:
            noise_paths.extend(glob.glob(os.path.join(nd, '**', '*.wav'), recursive=True))
            noise_paths.extend(glob.glob(os.path.join(nd, '**', '*.flac'), recursive=True))
    else:
        noise_paths = glob.glob(os.path.join(noise_dirs, '**', '*.wav'), recursive=True)

    # Exclude MUSAN speech subset to prevent target cancellation.
    noise_paths = [p for p in noise_paths if 'speech' not in p.replace('\\', '/').lower().split('/')]
    
    # Prefer 48k DEMAND samples over 16k duplicates.
    demand_paths = [p for p in noise_paths if 'demand' in p.replace('\\', '/').lower()]
    if demand_paths:
        # Check if 48k files exist; if so, drop the 16k ones
        has_48k = any('_48k' in p.replace('\\', '/').lower() for p in demand_paths)
        if has_48k:
            noise_paths = [
                p for p in noise_paths 
                if 'demand' not in p.replace('\\', '/').lower() or '_48k' in p.replace('\\', '/').lower()
            ]
            print("DEMAND dataset: Using only 48k files for maximum quality.")

    print(f"Found {len(clean_paths)} clean audio files.")
    print(f"Found {len(noise_paths)} noise audio files (speech excluded, DEMAND 48k preferred).")
    
    if not clean_paths or not noise_paths:
        raise ValueError("Audio directories are empty! Please check the dataset paths.")

    clean_ds = tf.data.Dataset.from_tensor_slices(clean_paths).shuffle(10000).repeat()
    noise_ds = tf.data.Dataset.from_tensor_slices(noise_paths).shuffle(10000).repeat()

    def load_and_mix(clean_path, noise_path):
        # Use separate read functions: clean gets RIR echo, noise does not
        clean_audio = tf.numpy_function(py_read_clean_audio, [clean_path], tf.float32)
        noise_audio = tf.numpy_function(py_read_noise_audio, [noise_path], tf.float32)
        
        clean_audio.set_shape([SAMPLES_PER_FRAGMENT])
        noise_audio.set_shape([SAMPLES_PER_FRAGMENT])
        
        # Training SNR range for extreme noise robustness
        snr = tf.random.uniform([], -15.0, 10.0)
        return mix_audio(clean_audio, noise_audio, snr)

    ds = tf.data.Dataset.zip((clean_ds, noise_ds))
    ds = ds.map(load_and_mix, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
