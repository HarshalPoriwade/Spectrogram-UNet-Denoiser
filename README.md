# Real-Time Edge Audio Denoiser (Spectrogram U-Net)

A highly optimized, edge-ready Deep Learning Audio Denoising engine built with TensorFlow. This model is capable of stripping extreme background noise (hiss, rumble, wind, chatter) from 48kHz high-fidelity audio streams in real-time, without distorting human speech.

## 🚀 Performance & Architecture

This project abandons standard 1D Wave-U-Net and LSTM approaches in favor of a **Frequency-Domain 2D Spectrogram U-Net** (6-layer deep bottleneck), ensuring no temporal memory loss and enabling native 48kHz full-spectrum audio processing.

*   **Model Size:** `17.1 MB` (Pure Float32, stateless `.tflite` format)
*   **Parameters:** `~4.48 Million` (Highly compressed bottleneck design)
*   **Speed:** `1.45x Real-Time` inference on standard edge CPUs (via XNNPACK Delegate). Latency < 10ms per block.
*   **Data Pipeline:** Dynamic `tf.data` pipeline mixing clean voice and noise on-the-fly at extreme Signal-to-Noise Ratios (`-15dB to +10dB`).

## 📊 Evaluation & Metrics

The model predicts an **Ideal Ratio Mask (IRM)** which is multiplied against the noisy spectrogram. The final finetuned model achieves state-of-the-art results on both synthetic and real-world benchmarks.

*   **Average SNR Improvement:** `+5.41 dB` across 5 extreme synthetic noise profiles.
*   **Peak SNR Improvement:** `+23.20 dB` on high-frequency static/hiss.
*   **Real-World Evaluation:** `36.22 dB` Pseudo-SNR on reference-less blind tests.
*   **DSP Optimization:** Custom 50% Overlap-Add algorithm utilizing a Hanning window to completely eliminate block-boundary clicking artifacts.

