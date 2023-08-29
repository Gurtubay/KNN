import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from collections import Counter
"""
k-Nearest Neighbors (KNN).

Variables:
train_data: Datos de entrenamiento
label_data: Etiquetas de entrenamiento
test_data: Datos de prueba
k: Número de vecinos más cercanos a considerar

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
        masComun = Counter(k_nearest_labels).masComun(1)
        pred_label.append(masComun[0][0])
        cont+=1
    
    return np.array(pred_label)

def accuracyScore(test_labels,pred_labels):
    correct = np.sum(test_labels == pred_labels)
    n = len(test_labels)
    acc = correct / n 
    return acc

"""
Calcula la distancia euclidiana entre dos puntos a y b.
"""
def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

data = pd.read_csv("Dataset.csv")
df = data.drop(columns=["id","Unnamed: 32"],axis=1)

df.groupby("diagnosis").size()
x = df.drop("diagnosis",axis=1).values
y = df["diagnosis"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=5)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

n = len(df)
k_max = math.sqrt(n)

normalAcu = [] 
k_value = range(1,24)

for i in k_value : 
    y_pred1 = knn_func(X_train,y_train,X_test,i)
    accur = accuracy_score(y_test, y_pred1)
    normalAcu.append(accur)

y_pred2 = knn_func(X_train,y_train,X_test,7)
acu=accuracyScore(y_test,y_pred2)
print('Precision!!')
print(acu*100)
