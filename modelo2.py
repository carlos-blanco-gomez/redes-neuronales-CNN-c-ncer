from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2

# Crear el modelo de CNN con modificaciones
model = Sequential()

# Primera capa convolucional con Batch Normalization
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa convolucional con Batch Normalization
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa convolucional opcional
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar y agregar capas densas con regularización
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid para clasificación binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Aumento de datos más avanzado
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    rotation_range=20,  # Rotación de imágenes
    width_shift_range=0.2,  # Desplazamiento horizontal
    height_shift_range=0.2  # Desplazamiento vertical
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
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('mejor_modelo.h5', monitor='val_loss', save_best_only=True)
]

# Entrenar el modelo con callbacks
model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size,
    callbacks=callbacks
)

# Guardar el modelo final
model.save('C:\\Users\\carlos\\Downloads\\projectIA\\proyecto_flask\\modelo\\cnn_model.h')
