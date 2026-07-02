import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Concatenate, UpSampling2D,
                                     BatchNormalization, Activation, Dropout)
from tensorflow.keras.models import Model

def create_spectrogram_unet():
    
    inputs = Input(shape=(128, 512, 1), name='input_spectrogram')
    
    # Layer 1: (128, 512, 1) -> (128, 256, 18)
    x1 = Conv2D(18, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    d1 = Conv2D(18, (3, 3), strides=(1, 2), padding='same', activation='relu')(x1)
    
    # Layer 2: (128, 256, 18) -> (64, 128, 36)
    x2 = Conv2D(36, (3, 3), padding='same')(d1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    d2 = Conv2D(36, (3, 3), strides=(2, 2), padding='same', activation='relu')(x2)
    
    # Layer 3: (64, 128, 36) -> (32, 64, 72)
    x3 = Conv2D(72, (3, 3), padding='same')(d2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    d3 = Conv2D(72, (3, 3), strides=(2, 2), padding='same', activation='relu')(x3)
    
    # Layer 4: (32, 64, 72) -> (16, 32, 144)
    x4 = Conv2D(144, (3, 3), padding='same')(d3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    d4 = Conv2D(144, (3, 3), strides=(2, 2), padding='same', activation='relu')(x4)
    
    # Layer 5: (16, 32, 144) -> (8, 16, 288)
    x5 = Conv2D(288, (3, 3), padding='same')(d4)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    d5 = Conv2D(288, (3, 3), strides=(2, 2), padding='same', activation='relu')(x5)
    
    # Bottleneck + Dropout
    bot = Conv2D(288, (3, 3), padding='same')(d5)
    bot = BatchNormalization()(bot)
    bot = Activation('relu')(bot)
    bot = Dropout(0.3)(bot)   
    
    # Decoder
    # Layer 5 Up: (8, 16) -> (16, 32)
    u5 = UpSampling2D(size=(2, 2))(bot)
    c5 = Concatenate()([u5, x5])
    y5 = Conv2D(288, (3, 3), padding='same', activation='relu')(c5)
    
    # Layer 4 Up: (16, 32) -> (32, 64)
    u4 = UpSampling2D(size=(2, 2))(y5)
    c4 = Concatenate()([u4, x4])
    y4 = Conv2D(144, (3, 3), padding='same', activation='relu')(c4)
    
    # Layer 3 Up: (32, 64) -> (64, 128)
    u3 = UpSampling2D(size=(2, 2))(y4)
    c3 = Concatenate()([u3, x3])
    y3 = Conv2D(72, (3, 3), padding='same', activation='relu')(c3)
    
    # Layer 2 Up: (64, 128) -> (128, 256)
    u2 = UpSampling2D(size=(2, 2))(y3)
    c2 = Concatenate()([u2, x2])
    y2 = Conv2D(36, (3, 3), padding='same', activation='relu')(c2)
    
    # Layer 1 Up: (128, 256) -> (128, 512)  [only upsample frequency, not time]
    u1 = UpSampling2D(size=(1, 2))(y2)
    c1 = Concatenate()([u1, x1])
    y1 = Conv2D(18, (3, 3), padding='same', activation='relu')(c1)
    
    # Output: Sigmoid gate (0.0 = erase this frequency, 1.0 = keep it)  
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(y1)
    
    return Model(inputs, outputs, name='SpectrogramUNet_48kHz')

if __name__ == "__main__":
    model = create_spectrogram_unet()
    model.summary()
    params = model.count_params()
    size_mb = (params * 4) / (1024 * 1024)
    print(f"\nTotal parameters : {params:,}")
    print(f"Float32 model size: {size_mb:.1f} MB")
