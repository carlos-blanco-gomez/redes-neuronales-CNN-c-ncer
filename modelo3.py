from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Crear el modelo de CNN
model = Sequential()

# Primer bloque de capas convolucional y de pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segundo bloque de capas convolucional y de pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercer bloque de capas convolucional y de pooling
model.add(Conv2D(128, (3, 3), activation='relu'))  # Capa adicional
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar las características y agregar capas densas
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Capas densas con más neuronas
model.add(Dropout(0.6))  # Aumentar el valor de dropout

# Capa de salida
model.add(Dense(1, activation='sigmoid'))  # Sigmoid para clasificación binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocesamiento de datos y aumento de datos
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
train_set = train_datagen.flow_from_directory(
    'C:\\Users\\carlos\\Downloads\\projectIA\\data\\train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'C:\\Users\\carlos\\Downloads\\projectIA\\data\\test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('mejor_modelo.h5', monitor='val_loss', save_best_only=True)

# Entrenar el modelo
model.fit(
    train_set,
    steps_per_epoch=len(train_set),
    epochs=10,
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stop, checkpoint]
)

# Guardar el modelo
model.save('C:\\Users\\carlos\\Downloads\\projectIA\\proyecto_flask\\modelo\\cnn_model.h5')

print("Modelo entrenado y guardado")
