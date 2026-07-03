import tensorflow as tf
from model import create_spectrogram_unet

def export():
    print("Building stateless Spectrogram 2D U-Net for inference...")
    model = create_spectrogram_unet()
    
    print("Loading weights from fine_spectrogram_unet_best.weights.h5...")
    try:
        model.load_weights('fine_spectrogram_unet_best.weights.h5')
    except Exception as e:
        print(f"Could not load weights. Have you trained the model yet? Error: {e}")
        return

    # 1. Export standard Float32 model (17.1 MB) for high-end phones
    print("Converting to TFLite (Pure Float32 for Perfect Studio Quality)...")
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_f32 = converter_f32.convert()
    
    with open('fine_spectrogram_unet_float32.tflite', 'wb') as f:
        f.write(tflite_model_f32)
    print("Saved -> fine_spectrogram_unet_float32.tflite (approx 17.1MB)")

    print("\nExport complete! (Float32)")

if __name__ == "__main__":
    export()
