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
k-Nearest Neighbors (KNN)

Variables:
train_data: Datos de entrenamiento
label_data: Etiquetas de entrenamiento
test_data: Datos de prueba
k: Número de vecinos mas cercanos a considerar

Retorn:
pred_label: Etiquetas predichas para los datos de prueba
"""
def knn_func(train_data, label_data, test_data, k):

    pred_label = []
    cont=0
    # Iterar sobre cada punto en el conjunto de prueba
    for test_point in test_data:
        #print(cont)
        #print(len(test_data))
        distances = [distancia_euclidiana(test_point, train_point) for train_point in train_data]  # Calcular distancia entre el punto de prueba (test_point) y todos los puntos en el conjunto de entrenamiento
        # Obtener los indices de los k neihbours mas cercanos
        k_indices = np.argsort(distances)[:k]
        
        #Obtener las etiquetas de los k neihbours más cercanos
        k_nearest_labels = [label_data[i] for i in k_indices]
        
        #Realizar una seleccion para encontrar la etiqueta mas comun
        masComun = Counter(k_nearest_labels).most_common(1)
        pred_label.append(masComun[0][0])
        cont+=1
    
    return np.array(pred_label)

#Funcion para calcular las metricas faltantes con la libreria de scikit-learn metrics
def calcular_metricas(test, pred):
    # Calcula la matriz de confusión
    matriz = confusion_matrix(test, pred)
    
    # Calcula el recall
    rec = recall_score(test, pred)
    
    # Calcula el f1 score
    f1 = f1_score(test, pred)
    
    return matriz, rec, f1

"""
Calcula la distancia euclidiana entre dos puntos a y b
"""
def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

"""
Se obtiene el indice de normalAcu para saber cual es el mejor valor de k
"""
# Definir la función que devuelve el índice del valor más grande en una lista
def best_k_value(list):
    best_acu = list[0]  # Inicializar con el primer elemento de la lista
    best_k = 0  # Inicializar el índice del valor máximo
    
    # Iterar a través de la lista para encontrar el valor máximo y su índice
    for i in range(1, len(list)):
        if list[i] > best_acu:
            best_acu = list[i]
            best_k = i
    
    return best_k

# Definimos una función para realizar la validación cruzada en función de k
def cross_val_knn(k, X_data, y_data):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_data, y_data, cv=10, scoring='accuracy') # 10-fold cross-validation
    return scores.mean()

def knn_func1(train_data,label_data,test_data,k) : 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data,label_data)
    pred_label = knn.predict(test_data)
    return pred_label

data = pd.read_csv("Dataset.csv") #Se lee el conjunto de datos "Dataset.csv" usando pandas

df = data.drop(columns=["id","Unnamed: 32"],axis=1)#Se eliminan las columnas no necesarias
df.groupby("diagnosis").size()

#Las caracteristicas y etiquetas se almacenan en x y y
x = df.drop("diagnosis",axis=1).values
y = df["diagnosis"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) #Las etiquetas se transforman a enteros usando LabelEncoder


#Los datos se dividen en conjuntos de entrenamiento y prueba
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.6,random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


X_temp, X_val, y_temp, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

n = len(df) #n es el numero de registros
k_max = math.sqrt(n)#Se define el valor maximo de k

normalAcu = [] #Se inicializa una lista para guardar la precision de los entrenamientos con diferentes k
k_value = range(1,math.ceil(k_max))#Se inicializan los posibles valores de k que van desde 1 hasta k_max

# Probamos diferentes valores de k
cross_val_scores = []

for i in k_value:
    cross_val_scores.append(cross_val_knn(i, X_train, y_train))

# Graficamos la precisión promedio de validación cruzada en función de k
plt.figure(figsize=(14, 7))
plt.plot(k_value, cross_val_scores, marker='o', linestyle='-')
plt.title('Precisión Promedio de Validación Cruzada vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Precisión Promedio')
plt.grid(True)
plt.show()

# Obtenemos el mejor k basado en la validación cruzada
best_k_cross_val = k_value[np.argmax(cross_val_scores)]
best_k_cross_val, max(cross_val_scores)

#Se realiza una búsqueda para encontrar el mejor valor de k probando diferentes valores y almacenando sus precisines
for i in k_value:
    #y_pred1 = knn_func(X_train,y_train,X_test,i) #Esto ejectura el modelo de knn con lso diferentes valores de k
    y_pred1 = knn_func(X_train,y_train,X_test,i)
    accur = accuracy_score(y_test, y_pred1)#Esto evalua la precision
    normalAcu.append(accur)#Aqui se guarda la precision

# Graficando la precisión en función del valor de k
plt.figure(figsize=(12, 6))
plt.plot(k_value, normalAcu, marker='o', linestyle='-')
plt.title('Precisión vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.grid(True)
plt.show()

# Graficar los errores de entrenamiento y validación en función de k
plt.figure(figsize=(14, 7))
plt.plot(k_value, train_errors, marker='o', label='Entrenamiento', linestyle='-')
plt.plot(k_value, validation_errors, marker='x', label='Validación', linestyle='-')
plt.title('Errores de Entrenamiento y Validación vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

best_k=best_k_value(normalAcu)#Aqui llama a la funcion que regresa el indice del mejor valor de k en la lista normalAcu

y_pred2 = knn_func1(X_train,y_train,X_test,best_k+1) #Aqui se vuelve a correr el modelo de knn con el valor mas adecuado de k
acu=accuracy_score(y_test,y_pred2)#Se obtiene la precision del modelo con el mejor valor de k

matriz_conf, rec_score, f1_sc = calcular_metricas(y_test, y_pred2)#Se llama a la funcion que obtiene las metricas faltantes
#Se imprimen las metricas
print('Matriz de confusion')
print(matriz_conf)
print('*-----*-----*-----*-----*-----*-----*')
print('Sensibilidad (recall)')
print(rec_score)
print('*-----*-----*-----*-----*-----*-----*')
print('F1-Score')
print(f1_sc)
print('*-----*-----*-----*-----*-----*-----*')
print('Precision!!')
print(acu*100)
