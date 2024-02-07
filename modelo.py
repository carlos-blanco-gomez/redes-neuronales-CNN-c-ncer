from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Crear el modelo de CNN
model = Sequential()

# Agregar la primera capa convolucional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Agregar la segunda capa convolucional
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar las características y agregar capas densas
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Usa 'softmax' para clasificación multiclase

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preprocesamiento de datos y aumento de datos
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
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

# Entrenar el modelo
model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size
)

# Guardar el modelo
model.save('C:\\Users\\carlos\\Downloads\\projectIA\\proyecto_flask\\modelo\\cnn_model.h5')
