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

def knn_func(train_data, label_data, test_data, k):
    pred_label = []
    for test_point in test_data:
        distances = [distancia_euclidiana(test_point, train_point) for train_point in train_data]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [label_data[i] for i in k_indices]
        masComun = Counter(k_nearest_labels).most_common(1)
        pred_label.append(masComun[0][0])
    return np.array(pred_label)

def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def calcular_metricas(test, pred):
    matriz = confusion_matrix(test, pred)    
    rec = recall_score(test, pred)    
    f1 = f1_score(test, pred)
    return matriz, rec, f1

def best_k_value(list):
    return np.argmax(list) + 1

def cross_val_knn(k, X_data, y_data):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_data, y_data, cv=10, scoring='accuracy')
    return scores.mean()

def knn_func1(train_data, label_data, test_data, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, label_data)
    pred_label = knn.predict(test_data)
    return pred_label

data = pd.read_csv("Dataset.csv")
df = data.drop(columns=["id", "Unnamed: 32"])
x = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_temp, X_val, y_temp, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
n = len(df)
k_max = math.sqrt(n)
k_value = range(1, math.ceil(k_max))

cross_val_scores = []
for i in k_value:
    cross_val_scores.append(cross_val_knn(i, X_train, y_train))

plt.figure(figsize=(14, 7))
plt.plot(k_value, cross_val_scores, marker='o', linestyle='-')
plt.title('Precisión Promedio de Validación Cruzada vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Precisión Promedio')
plt.grid(True)
plt.show()

best_k = best_k_value(cross_val_scores)
print(best_k)
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
final_predictions = final_knn.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
matriz_conf, rec_score, f1_sc = calcular_metricas(y_test, final_predictions)

print('Matriz de Confusión:', matriz_conf)
print('Sensibilidad (Recall):', rec_score)
print('F1-Score:', f1_sc)
print('Precisión:', final_accuracy)
