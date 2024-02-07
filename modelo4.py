from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2

# Crear el modelo de CNN con regularización L2
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos de entrenamiento y prueba
train_set = train_datagen.flow_from_directory('C:\\Users\\carlos\\Downloads\\projectIA\\data\\train', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\carlos\\Downloads\\projectIA\\data\\test', target_size=(64, 64), batch_size=32, class_mode='binary')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('mejor_modelo.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Entrenar el modelo
model.fit(
    train_set,
    steps_per_epoch=len(train_set),
    epochs=30,  # Aumentar el número de épocas
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Guardar el modelo
model.save('C:\\Users\\carlos\\Downloads\\projectIA\\proyecto_flask\\modelo\\cnn_model.h5')

print("Modelo entrenado y guardado como 'modelo_final.h5'")
