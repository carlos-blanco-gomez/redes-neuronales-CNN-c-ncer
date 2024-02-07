import os
import shutil

# Rutas de las carpetas origen y destino
ruta_origen = "C:\\Users\\carlos\\Downloads\\projectIA\\data\\train\\no_cancer"
ruta_destino = "C:\\Users\\carlos\\Downloads\\projectIA\\data\\test\\no_cancer"

# Lista de archivos en la carpeta origen
archivos = os.listdir(ruta_origen)

# Número de imágenes a mover
numero_imagenes = min(1000, len(archivos))

# Contador para las imágenes movidas
imagenes_movidas = 0

# Mover imágenes
for nombre_archivo in archivos[:numero_imagenes]:
    ruta_archivo_origen = os.path.join(ruta_origen, nombre_archivo)
    ruta_archivo_destino = os.path.join(ruta_destino, nombre_archivo)

    # Verificar si el archivo existe antes de moverlo
    if os.path.exists(ruta_archivo_origen):
        shutil.move(ruta_archivo_origen, ruta_archivo_destino)
        imagenes_movidas += 1
    else:
        print(f"Archivo no encontrado: {ruta_archivo_origen}")

print(f"Se han movido {imagenes_movidas} imágenes correctamente.")
