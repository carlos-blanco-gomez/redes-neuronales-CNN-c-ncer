from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ruta donde tienes guardado tu modelo de Keras
MODEL_PATH = 'C:\\Users\\carlos\\Downloads\\projectIA\\proyecto_flask\\modelo\\cnn_model.h5'
# Ruta de la carpeta donde se guardarán las imágenes subidas
UPLOAD_FOLDER = 'C:\\Users\\carlos\\Downloads\\projectIA\\proyecto_flask\\images'

# Asegúrate de que la carpeta de UPLOAD_FOLDER exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo de CNN
modelo = load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    data = {"success": False}
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)

            img = load_img(image_path, target_size=(64, 64))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = modelo.predict(img_array)
            prediction_label = 'Cáncer' if prediction[0][0] > 0.5 else 'No Cáncer'

            # Procesa el resultado para mostrarlo en un gráfico
            data["success"] = True
            data["prediction"] = prediction.tolist()
            data["prediction_label"] = prediction_label

            # Crear un gráfico con Matplotlib
            fig, ax = plt.subplots()
            ax.barh([0], [prediction[0][0]], color=['blue' if prediction_label == 'No Cáncer' else 'red'])
            ax.set_yticks([0])
            ax.set_yticklabels([prediction_label])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probabilidad')
            ax.set_title('Predicción de Cáncer')

            # Convertir el gráfico a una imagen para enviarlo al HTML
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            return render_template('index.html', plot_url=plot_url, data=data)

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
