# ✅ Sección I – Cargar el dataset “wine” de Scikit-learn

# Importar librerías
from sklearn.datasets import load_wine
import pandas as pd

# Cargar el dataset wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Mostrar las primeras filas
# df.head()
df.sample(10)  # Muestra 10 filas aleatorias, de cualquier clase

# Se importa el conjunto de datos wine que contiene información sobre vinos de 3 clases.
# Se crea un DataFrame de pandas con las características y la etiqueta (target), que representa el tipo de vino (0, 1 o 2).

# ✅ Sección II – Seleccionar las características deseadas

features = ['alcohol', 'magnesium', 'color_intensity']
df_selected = df[features + ['target']]

# df_selected.head()
df_selected.sample(10)

# Nos enfocamos en 3 características: alcohol, magnesium y color_intensity, como base para entrenar un clasificador.
# Estas características son numéricas y ayudan a diferenciar los tipos de vino.

# ✅ Sección III – Seleccionar dos clases para clasificación binaria

df_binary = df_selected[df_selected['target'].isin([0, 1])]

# Variables predictoras y objetivo
X = df_binary[features].values
y = df_binary['target'].values

# El dataset wine tiene 3 clases (0, 1, 2), pero necesitamos una clasificación binaria, así que seleccionamos solo las clases 0 y 1.
# X es la matriz de características, e y es el vector de etiquetas (0 o 1).

# ✅ Sección IV - Divida la data en dos subconjuntos: uno de entrenamiento y uno de prueba

from sklearn.model_selection import train_test_split

# División de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Se dividen los datos de entrenamiento manteniendo proporcionalidad

# ✅ Sección V - Usando los datos de entrenamiento, implemente un clasificador binario, basado en el algoritmo del Perceptrón Simple.


import numpy as np

# Convertir etiquetas de 0 y 1 a -1 y 1 para compatibilidad con el algoritmo
y_train_bin = np.where(y_train == 0, -1, 1)

# Inicializar pesos y sesgo
weights = np.zeros(X_train.shape[1])
bias = 0

# Hiperparámetros
learning_rate = 0.01
epochs = 100

# Entrenamiento
for epoch in range(epochs):
    for xi, target in zip(X_train, y_train_bin):
        activation = np.dot(xi, weights) + bias
        prediction = np.sign(activation)
        if prediction != target:
            weights += learning_rate * target * xi
            bias += learning_rate * target

# Utilizando el algoritmo de clasificación binaria supervisado, donde su fórmula es:
# y​=sign(wT⋅x+b) 

# donde:

# 𝑥 es el vector de características de entrada.

# 𝑤 son los pesos del modelo.

# 𝑏 es el sesgo (bias).

# sign ( ) devuelve +1 o -1 (nosotros adaptaremos a 0 o 1).

# Esto último porque el Perceptrón de Rosenblatt toma decisiones usando el signo de esta expresión:

# y​=sign(w⋅x+b) 

# Y luego actualiza los pesos solo si se equivoca usando esta regla:

# w←w+η⋅y⋅x 

# Donde:

# 𝑦 es la etiqueta real, que debe ser -1 o 1

# 𝜂 es la tasa de aprendizaje

# 𝑥 es el vector de características

# Si 𝑦 ∈ { 0 , 1 } y∈{0,1}, esta fórmula no funciona correctamente, porque por ejemplo:

# Si 𝑦 = 0 y=0, entonces 𝑦 ⋅ 𝑥 = 0 y⋅x=0, y los pesos no se actualizan nunca.


# ✅ Sección VI - Usando los datos de prueba, evalue el desempeño del algoritmo y muéstrelo a través de una matriz de confusión.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Convertir etiquetas de prueba a -1 y 1
y_test_bin = np.where(y_test == 0, -1, 1)

# Función de predicción
def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.where(linear_output >= 0, 1, -1)  # Aplicar función escalón

# Predecir sobre los datos de prueba
y_pred_bin = predict(X_test, weights, bias)

# Calcular y mostrar matriz de confusión
cm = confusion_matrix(y_test_bin, y_pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clase -1 (original 0)", "Clase 1 (original 1)"])
disp.plot()
plt.title("Matriz de Confusión - Perceptrón Manual")
plt.grid(False)
plt.show()

# Importación de librerías: Se importan las funciones confusion_matrix y ConfusionMatrixDisplay de sklearn.metrics para calcular y visualizar la matriz de confusión. También se importa matplotlib.pyplot para personalizar la visualización.

# Conversión de etiquetas: Como el Perceptrón clásico trabaja con etiquetas binarias -1 y 1, se transforman las etiquetas originales de prueba (0 y 1) en -1 y 1 respectivamente.

# Definición de la función de predicción: Se implementa la función predict, que aplica la regla del Perceptrón:

# Calcula la salida lineal: 𝑤 ⋅ 𝑥 + 𝑏

# Aplica la función escalón: devuelve 1 si la salida es mayor o igual que 0, y -1 en caso contrario.

# Predicciones sobre el conjunto de prueba: Se usa la función anterior junto con los pesos y el sesgo aprendidos durante el entrenamiento para predecir las etiquetas de los datos de prueba.

# Cálculo y visualización de la matriz de confusión: Se compara el vector de etiquetas verdaderas con el de predicciones, para generar una matriz de confusión. Luego, se visualiza gráficamente, indicando cuántas predicciones fueron correctas y cuántas incorrectas para cada clase.


# ✅ Sección VII - Usando los datos de prueba, evalue el desempeño del algoritmo y muéstrelo a través de una matriz de confusión.

from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Crear el modelo de Perceptrón
clf = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# Entrenar el modelo con los datos originales (etiquetas 0 y 1)
clf.fit(X_train, y_train)

# Realizar predicciones sobre los datos de prueba
y_pred_sklearn = clf.predict(X_test)

# Calcular y mostrar matriz de confusión
cm = confusion_matrix(y_test, y_pred_sklearn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clase 0", "Clase 1"])
disp.plot()
plt.title("Matriz de Confusión - Perceptrón Scikit-learn")
plt.grid(False)
plt.show()


# En esta sección se utiliza la clase Perceptron de scikit-learn, una implementación optimizada del algoritmo del Perceptrón para clasificación supervisada. A continuación se describen los pasos principales:

# Creación del modelo: Se inicializa un objeto de tipo Perceptron con:

# max_iter=1000: número máximo de iteraciones para el entrenamiento.

# eta0=0.1: tasa de aprendizaje.

# random_state=42: semilla para asegurar la reproducibilidad de los resultados.

# Entrenamiento: Se entrena el modelo utilizando X_train y y_train. A diferencia de nuestro Perceptrón manual, aquí no es necesario convertir las etiquetas a -1 y 1; scikit-learn puede trabajar directamente con 0 y 1.

# Predicción: Se usan los datos de prueba X_test para predecir las etiquetas correspondientes (y_pred_sklearn).

# Evaluación con matriz de confusión: Se calcula la matriz de confusión, que muestra cómo el modelo clasifica cada clase. Esto permite comparar visualmente su rendimiento con el Perceptrón manual.














