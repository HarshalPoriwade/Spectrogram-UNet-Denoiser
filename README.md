# Real-Time Edge Audio Denoiser (Spectrogram U-Net)

A highly optimized, edge-ready Deep Learning Audio Denoising engine built with TensorFlow. This model is capable of stripping extreme background noise (hiss, rumble, wind, chatter) from 48kHz high-fidelity audio streams in real-time, without distorting human speech.

## 🚀 Technical Architecture & Achievements

*   **Model Architecture:** Designed a 4.48M parameter Spectrogram U-Net (6-layer, 32→256 channels) delivering denoising quality comparable to 15M+ parameter models on native 48kHz audio via STFT with Log1p compression.
*   **Data Engineering:** Built a robust `tf.data` pipeline (16,000 slices/epoch) with on-the-fly noise mixing. Applied two-stage training on extreme SNR [-15dB to 10dB] with Dynamic Signal Normalization preserving Ideal Ratio Mask (IRM).
*   **Acoustic Performance:** Achieved +8.95dB SI-SDR improvement on studio speech pairs. Blind real-world evaluation scored 32–47dB Pseudo-SNR. Applied Stochastic Weight Averaging (SWA) for production model fusion.
*   **Edge Deployment:** Exported as a 17.1MB TFLite model achieving <10ms latency per block and 1.5x real-time CPU inference via the XNNPACK delegate.

