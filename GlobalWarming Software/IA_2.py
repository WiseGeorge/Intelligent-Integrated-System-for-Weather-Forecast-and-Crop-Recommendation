import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importando Datos
temperature_df = pd.read_excel("../Datasets/TempFilter.xlsx")

#Cargando Set Datos
x_train = temperature_df['Minima Media']
y_train = temperature_df['Maxima Media']

#Visualizacion
sns.scatterplot(x=x_train,y=y_train)

#Creando Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))

#Mostrando Modelo
model.summary()

#Compilado
model.compile(optimizer = tf.keras.optimizers.Adam(30), loss = 'mean_squared_error')

#Entrenando el Modelo
epochs_hist = model.fit(x_train, y_train, epochs = 50)

#Evaluando Modelo
epochs_hist.history.keys()

#Grafico
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Perdida durante Entrenamiento del Modelo')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend('Training Loss')
plt.show()

model.get_weights()

#Predicciones
min = 18
Prediction = model.predict([min])

print(Prediction)
