import numpy as np
import tensorflow as tf
import keras 

# Cargar el modelo guardado
model = tf.keras.models.load_model("modelo_normalizado.keras")

# Cargar los datos de normalización
mean_x, std_x, mean_y, std_y = np.load("normalization_params.npy")

# Solicitar el año al usuario
year = int(input("Ingrese el año que quiere predecir: "))

# Normalizar el año
year_normalizado = (year - mean_x) / std_x

# Convertir el año en un array de NumPy con formato adecuado para la predicción
year_normalizado = np.array([[year_normalizado]])

# Hacer la predicción
prediccion_normalizada = model.predict(year_normalizado)

# Desnormalizar la predicción para obtener la temperatura original
prediccion = prediccion_normalizada * std_y + mean_y

# Mostrar el resultado
print(f"Temperatura predicha para el año {year}: {prediccion[0][0]}")
