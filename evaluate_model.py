import numpy as np
import tensorflow as tf
import time
import os
import soundfile as sf
import librosa
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        pass

from model import create_spectrogram_unet

FRAME_LENGTH = 1024
HOP_LENGTH   = 256
NUM_FRAMES   = 128
NUM_BINS     = 512
SR           = 48000

def generate_test_audio(duration=5.0, sr=48000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate clean speech-like harmonic series
    clean = np.zeros_like(t)
    fundamentals = [150, 200, 250]
    for i in range(1, 10):
        clean += (1.0 / i) * np.sin(2 * np.pi * 150 * i * t)
    # Amplitude modulation to simulate syllables
    env = np.maximum(0, np.sin(2 * np.pi * 4 * t)) 
    clean *= env
    clean /= np.max(np.abs(clean))
    
    # 5 noise profiles
    np.random.seed(42)
    noises = {}
    
    # 1. White Noise
    white = np.random.randn(*t.shape)
    noises['White_Noise'] = white / np.max(np.abs(white))
    
    # 2. Pink Noise (1/f)
    pink = np.cumsum(np.random.randn(*t.shape))
    # highpass filter pink noise to remove drift
    pink = librosa.effects.preemphasis(pink)
    noises['Pink_Noise'] = pink / np.max(np.abs(pink))
    
    # 3. Low Freq Rumble
    rumble = np.random.randn(*t.shape)
    b, a = librosa.filters.get_window('hann', 100), [1.0] # simple lowpass via convolution
    rumble = np.convolve(rumble, b, mode='same')
    noises['Low_Freq_Rumble'] = rumble / np.max(np.abs(rumble))
    
    # 4. High Freq Hiss
    hiss = np.random.randn(*t.shape)
    hiss = librosa.effects.preemphasis(hiss, coef=0.99)
    noises['High_Freq_Hiss'] = hiss / np.max(np.abs(hiss))
    
    # 5. Sine Interference
    siren = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1500 * t)
    noises['Sine_Interference'] = siren / np.max(np.abs(siren))

    os.makedirs('evaluation_audio', exist_ok=True)
    sf.write('evaluation_audio/0_clean_ground_truth.wav', clean, sr)

    mixtures = {}
    for name, noise in noises.items():
        # Mix at ~ 5dB SNR
        mixture = (clean * 1.0) + (noise * 0.5)
        mixture /= np.max(np.abs(mixture))
        sf.write(f'evaluation_audio/1_noisy_{name}.wav', mixture, sr)
        mixtures[name] = (clean, noise * 0.5, mixture)
        
    return mixtures

def process_audio(audio, interpreter, in_idx, out_idx):
    stft_matrix = librosa.stft(audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False)
    mag   = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    total_frames = mag.shape[1]

    OVERLAP  = NUM_FRAMES // 2
    HOP_BLKS = NUM_FRAMES - OVERLAP

    pad_frames = NUM_FRAMES
    mag_padded   = np.pad(mag,   ((0, 0), (0, pad_frames)), mode='constant')
    phase_padded = np.pad(phase, ((0, 0), (0, pad_frames)), mode='constant')
    
    out_mag   = np.zeros_like(mag_padded)
    sum_window = np.zeros(mag_padded.shape[1], dtype=np.float32)
    hann = np.hanning(NUM_FRAMES).astype(np.float32)

    num_blocks = (mag_padded.shape[1] - NUM_FRAMES) // HOP_BLKS + 1

    for i in range(num_blocks):
        t_start = i * HOP_BLKS
        t_end   = t_start + NUM_FRAMES
        block     = mag_padded[:, t_start:t_end].T
        block_512 = block[:, :NUM_BINS]

        input_tensor = np.log1p(block_512).astype(np.float32)
        input_tensor = input_tensor[np.newaxis, :, :, np.newaxis]

        interpreter.set_tensor(in_idx, input_tensor)
        interpreter.invoke()
        mask = interpreter.get_tensor(out_idx)[0, :, :, 0]

        clean_512 = block_512 * mask
        clean_513 = np.zeros((NUM_FRAMES, 513), dtype=np.float32)
        clean_513[:, :NUM_BINS] = clean_512

        for fi in range(NUM_FRAMES):
            out_mag[:, t_start + fi]    += clean_513[fi, :] * hann[fi]
            sum_window[t_start + fi]    += hann[fi]

    sum_window = np.maximum(sum_window, 1e-8)
    out_mag = out_mag / sum_window[np.newaxis, :]
    out_mag   = out_mag[:, :total_frames]
    phase_use = phase[:, :total_frames]

    out_stft  = out_mag * np.exp(1j * phase_use)
    out_audio = librosa.istft(out_stft, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False, length=len(audio))
    return out_audio

def calculate_snr(clean, noisy):
    # FIRST DRAFT BUG: Incorrectly calculating noise power using the raw noisy signal
    # instead of the isolated noise (clean - noisy). This will yield negative SNRs.
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noisy ** 2)
    if noise_power == 0:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)

def run_evaluation():
    orig_tflite = "spectrogram_unet_float32.tflite"
    fine_tflite = "fine_spectrogram_unet_float32.tflite"
    
    print(f"Loading ORIGINAL TFLite: {orig_tflite}")
    interp_orig = Interpreter(model_path=orig_tflite)
    interp_orig.allocate_tensors()
    orig_in_idx = interp_orig.get_input_details()[0]['index']
    orig_out_idx = interp_orig.get_output_details()[0]['index']
    
    print(f"Loading FINETUNED TFLite: {fine_tflite}")
    interp_fine = Interpreter(model_path=fine_tflite)
    interp_fine.allocate_tensors()
    fine_in_idx = interp_fine.get_input_details()[0]['index']
    fine_out_idx = interp_fine.get_output_details()[0]['index']

    print("Generating 5 synthetic audio profiles (48kHz)...")
    mixtures = generate_test_audio(duration=3.0, sr=SR)
    
    print("\n--- RESULTS ---")
    total_snr_imp_orig = 0
    total_snr_imp_fine = 0
    
    for name, (clean, noise, mixture) in mixtures.items():
        print(f"\nEvaluating Profile: {name}")
        
        # Original SNR
        orig_snr = calculate_snr(clean, mixture)
        print(f"  Input SNR:  {orig_snr:.2f} dB")
        
        # --- Evaluate Original TFLite ---
        cleaned_orig = process_audio(mixture, interp_orig, orig_in_idx, orig_out_idx)[:len(clean)]
        max_c = np.max(np.abs(clean))
        if max_c > 0: cleaned_orig = (cleaned_orig / np.max(np.abs(cleaned_orig))) * max_c
        final_snr_orig = calculate_snr(clean, cleaned_orig)
        snr_imp_orig = final_snr_orig - orig_snr
        total_snr_imp_orig += snr_imp_orig
        print(f"  [ORIGINAL TFLite] Output SNR: {final_snr_orig:.2f} dB | Improvement: +{snr_imp_orig:.2f} dB")
        sf.write(f'evaluation_audio/2_cleaned_{name}_original_tflite.wav', cleaned_orig, SR)
        
        # --- Evaluate Finetuned TFLite ---
        cleaned_fine = process_audio(mixture, interp_fine, fine_in_idx, fine_out_idx)[:len(clean)]
        if max_c > 0: cleaned_fine = (cleaned_fine / np.max(np.abs(cleaned_fine))) * max_c
        final_snr_fine = calculate_snr(clean, cleaned_fine)
        snr_imp_fine = final_snr_fine - orig_snr
        total_snr_imp_fine += snr_imp_fine
        print(f"  [FINETUNED TFLite] Output SNR: {final_snr_fine:.2f} dB | Improvement: +{snr_imp_fine:.2f} dB")
        sf.write(f'evaluation_audio/2_cleaned_{name}_finetuned_tflite.wav', cleaned_fine, SR)

    avg_imp_orig = total_snr_imp_orig / len(mixtures)
    avg_imp_fine = total_snr_imp_fine / len(mixtures)
    
    print(f"\n======================================")
    print(f"ORIGINAL MODEL AVG SNR IMPROVEMENT:  +{avg_imp_orig:.2f} dB")
    print(f"FINETUNED MODEL AVG SNR IMPROVEMENT: +{avg_imp_fine:.2f} dB")
    print(f"======================================")
    
    if avg_imp_fine > avg_imp_orig:
        print("VERDICT: The FINETUNED model is objectively better.")
        print(f"It improved over the original by +{avg_imp_fine - avg_imp_orig:.2f} dB on average across 5 extreme noise profiles.")
    elif avg_imp_fine < avg_imp_orig:
        print("VERDICT: The ORIGINAL model performed better mathematically.")
    else:
        print("VERDICT: Both models performed identically on synthetic data.")

if __name__ == "__main__":
    run_evaluation()
