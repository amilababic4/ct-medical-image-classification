from tensorflow.keras import layers, models
import tensorflow as tf

def build_simple_3d_cnn():
    model = models.Sequential([
        layers.Conv3D(16, (3,3,3), 
                      activation='relu', 
                      padding='same',
                      input_shape=(28,28,28,1)),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2,2,2)),
        layers.Dropout(0.25),
        
        layers.Conv3D(32, (3,3,3), 
                      activation='relu', 
                      padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2,2,2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model