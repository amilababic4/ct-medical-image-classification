def build_resnet3d_no_bn():
    inputs = layers.Input(shape=(28,28,28,1))
    
    x = layers.Conv3D(32, (3,3,3), 
                      padding='same',
                      activation='relu')(inputs)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Residual blok 1
    shortcut1 = x
    x = layers.Conv3D(32, (3,3,3), 
                      padding='same',
                      activation='relu')(x)
    x = layers.Conv3D(32, (3,3,3), 
                      padding='same')(x)
    x = layers.Add()([x, shortcut1])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    
    # Residual blok 2
    shortcut2 = layers.Conv3D(64, (1,1,1), 
                               padding='same')(x)
    x = layers.Conv3D(64, (3,3,3), 
                      padding='same',
                      activation='relu')(x)
    x = layers.Conv3D(64, (3,3,3), 
                      padding='same')(x)
    x = layers.Add()([x, shortcut2])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model