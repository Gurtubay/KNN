import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from collections import Counter

"""
Funcion para realizar la clasificación KNN

Parametros:
-train_data: Datos de entrenamiento
-label_data: Etiquetas de entrenamiento
-test_data: Datos de prueba
-k: Número de vecinos a considerar.

Retorn:
-pred_label: Etiquetas predichas para los datos de prueba
"""

def knn_func(train_data, label_data, test_data, k):

    # Crear un objeto de clasificador KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenar el modelo con los datos de entrenamiento
    knn.fit(train_data, label_data)
    
    # Predecir las etiquetas para los datos de prueba
    pred_label = knn.predict(test_data)
    
    return pred_label

"""
Función para calcular la precisión promedio usando validación cruzada.

Parametros:
- k: Numero de vecinos.
- X_data: Datos de entrada
- y_data: Etiquetas

Retorn:
-Precision promedio de validacion cruzada
"""
def cross_val_knn(k, X_data, y_data):

    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_data, y_data, cv=10, scoring='accuracy')
    return scores.mean()

"""
Función para calcular la distancia euclidiana entre dos puntos

Parametros:
- a, b: Puntos a comparar

Retorn:
- Distancia euclidiana entre a y b
"""
def distancia_euclidiana(a, b):

    return np.sqrt(np.sum((a - b) ** 2))

"""
Funcion para calcular metricas de desempeño.

Parametros:
-test: Etiquetas reales
-pred: Etiquetas predichas

Retorn:
-matriz: Matriz de confusion.
-rec: Sensibilidad (Recall).
-f1: F1-Score.
"""
def calcular_metricas(test, pred):

    matriz = confusion_matrix(test, pred)    
    rec = recall_score(test, pred)    
    f1 = f1_score(test, pred)
    return matriz, rec, f1

"""
Funcion para encontrar el mejor valor de k basado en la precision

Parametros:
- list: Lista de precisión para diferentes valores de k.

Retorn:
- best_k: Mejor valor de k.
"""
def best_k_value(list):

    best_acu = list[0]
    best_k = 0  
    for i in range(1, len(list)):
        if list[i] > best_acu:
            best_acu = list[i]
            best_k = i
    return best_k

data = pd.read_csv("Dataset.csv")  # Modified path to read the dataset

df = data.drop(columns=["id", "Unnamed: 32"], axis=1)
df.groupby("diagnosis").size()

# Features and labels are stored in x and y
x = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Labels are transformed to integers using LabelEncoder

# Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# X_temp, X_val, y_temp, y_val = train_test_split(X_train, y_train, test_size=0.6, random_state=42)

# Splitting the data into training, validation, and testing sets with a 60/20/20 split
X_temp, X_test, y_temp, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Fitting the scaler only with training data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Transforming the validation and test data using the same scaler
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train.shape, X_val.shape, X_test.shape

# Calculating proportions for the pie charts
train_size = len(X_train)
test_size = len(X_test)
temp_size = len(X_temp)
val_size = len(X_val)


n = len(df)  # n is the number of records
k_max = math.sqrt(n)  # The maximum value of k is defined

k_value = range(1, math.ceil(k_max))  # Possible values of k are initialized from 1 to k_max

# Compute the training and validation errors for each k
train_errors = []
validation_errors = []

for i in k_value:
    y_pred_train = knn_func(X_train, y_train, X_temp, i)  # Training predictions
    y_pred_val = knn_func(X_train, y_train, X_val, i)  # Validation predictions
    
    train_error = 1 - accuracy_score(y_temp, y_pred_train)
    val_error = 1 - accuracy_score(y_val, y_pred_val)
    
    train_errors.append(train_error)
    validation_errors.append(val_error)

# Pie chart for initial train-test split
labels = ['Entrenamiento', 'Prueba']
sizes = [train_size, test_size]
colors = ['gold', 'blue']
explode = (0.1, 0)  # explode 1st slice for emphasis

# Plotting the pie chart for train-test split
plt.figure(figsize=(10, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Division inicial entre datos de entrenamiento y prueba')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Pie chart for train-validation split from the training data
labels = ['Entrenamiento', 'Validacion']
sizes = [temp_size, val_size]
colors = ['yellowgreen', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice for emphasis

# Plotting the pie chart for train-validation split
plt.figure(figsize=(10, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Division entre datos de entrenamiento y validacion')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Plot the training and validation errors against k values
plt.figure(figsize=(14, 7))
plt.plot(k_value, train_errors, marker='o', label='Entrenamiento', linestyle='-')
plt.plot(k_value, validation_errors, marker='x', label='Validacion', linestyle='-')
plt.title('Errores de Entrenamiento y Validacion vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

cross_val_scores = []
for i in k_value:
    cross_val_scores.append(cross_val_knn(i, X_train, y_train))

# Graficamos la precisión promedio de validacion cruzada en función de k
plt.figure(figsize=(14, 7))
plt.plot(k_value, cross_val_scores, marker='o', linestyle='-')
plt.title('Precisión Promedio de Validacion Cruzada vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Precision Promedio')
plt.grid(True)
plt.show()

best_k = best_k_value(cross_val_scores)
print(best_k)
final_knn = KNeighborsClassifier(n_neighbors=5) #Para obtener una mejora en el modelo cambiar best_k por 5
final_knn.fit(X_train, y_train)
final_predictions = final_knn.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
matriz_conf, rec_score, f1_sc = calcular_metricas(y_test, final_predictions)


print('Matriz de Confusion:', matriz_conf)
print('Sensibilidad (Recall):', rec_score)
print('F1-Score:', f1_sc)
print('Precision:', final_accuracy)