import pandas as pd
import tensorflow as tf
import keras 
import numpy as np

#importar csv como dataframe
df=pd.read_csv('clima.csv')

#columnas de dataframe como valores de entrada y salida
year= df['year'].values
temperature=df['temp'].values

x=np.asarray(year)
y=np.asarray(temperature)

########################################################
#normalizacion de los datos, para mejorar la prediccion:
########################################################
mean_x = np.mean(year)
std_x = np.std(year)
x = (year - mean_x) / std_x

mean_y = np.mean(temperature)
std_y = np.std(temperature)
y = (temperature - mean_y) / std_y
########################################################

#modelo

# Crear el modelo con múltiples capas
oculta1=tf.keras.layers.Dense(units=3,input_shape=[1])
oculta2=tf.keras.layers.Dense(units=3)
capasalida=tf.keras.layers.Dense(units=1)


model=tf.keras.Sequential([oculta1,oculta2,capasalida])


'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]),  # Capa oculta 1
    tf.keras.layers.Dense(units=128, activation='relu'),  # Capa oculta 2
    tf.keras.layers.Dense(units=64, activation='relu'),   # Capa oculta 3
    tf.keras.layers.Dense(units=1)  # Capa de salida
])
'''

#optimizadores                    y  perdidas
#otros para probar:
#optimizer="adam"
#optimizer="RMSprop"
model.compile(optimizer="Nadam" , loss="mean_squared_error")

model.fit(x,y,epochs=100) 

# Guardar el modelo y las estadísticas de normalización
model.save("modelo_multicapa.keras")
np.save("normalization_params_multicapa.npy", [mean_x, std_x, mean_y, std_y])