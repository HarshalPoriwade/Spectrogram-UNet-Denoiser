import tensorflow as tf
from dataset import create_tf_dataset
from model import create_spectrogram_unet
import os
import time

FRAME_LENGTH = 1024
HOP_LENGTH   = 256
NUM_FRAMES   = 128
NUM_BINS     = 512   # Trimmed from 513 (drop Nyquist bin, keep power-of-2)

def prepare_spectrogram(mixed, clean):
    """
    Converts a pair of raw audio signals into (noisy_spectrogram, ideal_mask).
    """
    # Compute STFT for both signals
    mixed_stft = tf.signal.stft(mixed, frame_length=FRAME_LENGTH, frame_step=HOP_LENGTH)
    clean_stft = tf.signal.stft(clean, frame_length=FRAME_LENGTH, frame_step=HOP_LENGTH)
    
    # Get magnitude spectrogram and trim Nyquist bin
    mixed_mag = tf.abs(mixed_stft)[:, :NUM_FRAMES, :NUM_BINS]
    clean_mag = tf.abs(clean_stft)[:, :NUM_FRAMES, :NUM_BINS]
    
    # Generate Ideal Ratio Mask (IRM)
    mask = clean_mag / (mixed_mag + 1e-8)
    mask = tf.clip_by_value(mask, 0.0, 1.0)
    
    # Log-compress input magnitude
    mixed_log = tf.math.log1p(mixed_mag)
    
    # 5. Add channel dim for Conv2D: (Batch, 128, 512) -> (Batch, 128, 512, 1)
    mixed_log = tf.expand_dims(mixed_log, -1)
    mask      = tf.expand_dims(mask, -1)
    
    # 6. Explicitly set static shapes so Keras can build the computation graph
    mixed_log.set_shape([None, NUM_FRAMES, NUM_BINS, 1])
    mask.set_shape([None, NUM_FRAMES, NUM_BINS, 1])
    
    return mixed_log, mask

def spectral_convergence_loss(y_true_mag, y_pred_mag):
    """
    Measures how well the predicted spectrogram's *structure* matches the clean one.
    Combined with MAE, this prevents the 'muffled but noise-free' failure mode.
    """
    sc = tf.norm(y_true_mag - y_pred_mag, ord='fro', axis=[-2,-1]) / (tf.norm(y_true_mag, ord='fro', axis=[-2,-1]) + 1e-8)
    return tf.reduce_mean(sc)

def combined_loss(y_true, y_pred):
    """
    Production-grade combined loss:
    - MAE on the mask          : penalizes large, obvious mask errors
    - Spectral Convergence     : ensures the output spectrogram structure is correct
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    sc  = spectral_convergence_loss(
        tf.squeeze(y_true, -1),
        tf.squeeze(y_pred, -1)
    )
    return mae + 0.1 * sc

class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_hours=11.0):
        super().__init__()
        self.max_seconds = max_hours * 3600
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"\n[TimeLimitCallback] {self.max_seconds/3600:.1f}h reached. Stopping safely.")
            self.model.stop_training = True

def train():
    CLEAN_DIR = '/kaggle/input/datasets/pratt3000/vctk-corpus'
    RIR_DIR   = '/kaggle/input/datasets/tuannguyenvananh/room-impulse-response-and-noise-database'
    NOISE_DIRS = [
        '/kaggle/input/datasets/chrisfilo/demand',
        '/kaggle/input/datasets/eliasmarcon/environmental-sound-classification-50',
        '/kaggle/input/datasets/nhattruongdev/musan-noise',
        '/kaggle/input/datasets/chrisfilo/urbansound8k',
    ]

    # Spectrogram models are memory-efficient. Batch 16 is safe on Kaggle P100/T4.
    BATCH_SIZE = 16

    print("\n=== Loading Dataset ===")
    dataset = create_tf_dataset(CLEAN_DIR, NOISE_DIRS, batch_size=BATCH_SIZE, rir_dir=RIR_DIR)

    # Map raw audio pairs -> (noisy spectrogram, ideal mask) on the GPU
    dataset = dataset.map(prepare_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)

    print("\n=== Building Model ===")
    model = create_spectrogram_unet()
    model.summary()

    # LR=3e-4 (not 1e-3!) is the sweet spot for mask prediction.
    # 3e-4 prevents mask saturation at 0.5
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=optimizer, loss=combined_loss)

    checkpoint_path = 'spectrogram_unet_best.weights.h5'
    
    # Basic local loading only
    if os.path.exists(checkpoint_path):
        print(f"\nResuming from local working directory: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    else:
        print("\nNo checkpoint found. Starting fresh.")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    print("\n=== Starting Training (max 150 epochs or 11 hours) ===")
    model.fit(
        dataset,
        steps_per_epoch=1000,   # 1000 x 16 samples = 16,000 audio slices / epoch
        epochs=150,
        callbacks=callbacks
    )

    model.save_weights('spectrogram_unet_final.weights.h5')
    print("\nTraining complete!")
    print("  Best weights  -> spectrogram_unet_best.weights.h5")
    print("  Final weights -> spectrogram_unet_final.weights.h5")
    print("\nNext step: run export_tflite.py to generate the 17.1MB TFLite model.")

if __name__ == "__main__":
    train()
