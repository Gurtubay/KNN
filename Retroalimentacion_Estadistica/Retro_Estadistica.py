import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

#Guarda en variable df sin la columna price
X = df.drop(columns=['price'])

#Se escalan y transforman el df
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)

# Crear un pipeline
pipe = Pipeline([
    ('select_best', SelectKBest(score_func=f_regression, k='all'))  # Prueba F para seleccionar características
])

#Implementar el pipeline al data frame
pipe.fit(df_scaled, df['price'])

# Obtener los puntajes de las características seleccionadas por SelectKBest
feature_scores = pipe.named_steps['select_best'].scores_

# Obtener las características seleccionadas automáticamente (las mejores)
selected_features = X.columns[pipe.named_steps['select_best'].get_support()]

# Imprimir las características seleccionadas y sus puntajes
print("Características seleccionadas automáticamente:")
print(selected_features)
print("Puntajes de las características:")
print(feature_scores)

# Seleccionar la variable objetivo basada en el puntaje de correlación más alto
obj = selected_features[np.argmax(feature_scores)]

# Imprimir la variable objetivo seleccionada automáticamente
print("Variable objetivo seleccionada automáticamente:", obj)

#seleccionar el objetivo con base en la correlacion
obj = selected_features[np.argmax(feature_scores)]
car = selected_features.tolist()

# Dividir los datos en conjunto de entrenamiento y prueba
X = df[car]
y = df[obj]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Se entrena el modelo con base en al division anterior
model = LinearRegression()
model.fit(X_train, y_train)

#Se realizan pruebas al predecir con la division de prueba
y_predict = model.predict(X_test)

#Metricas de resultados
r2 = r2_score(y_test, y_predict)
msError = mean_squared_error(y_test, y_predict)
rmsError = np.sqrt(msError)


print("R cuadrado=", r2)
print("Error de promedio cuadrado=", msError)
print("Error de la raiz del promedio cuadrado=", rmsError)

# Cross validation para ver los valores mas optimos del modelo
cross_val= cross_val_score(model, X, y, cv=5, scoring='r2')
print("Puntajes del cross validation=", cross_val)

#Regularizar el modelo
model_reg = Ridge(alpha=1.0)
model_reg.fit(X_train, y_train)
y_predict_reg = model_reg.predict(X_test)
r2_reg = r2_score(y_test, y_predict_reg)
print("R cuadrada regularizad = ", r2_reg)

#Escalamiento de los datos
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)

#Reduccion de dimensionalidad
pca = PCA(n_components=5)#Parametro (Numero de componentes)
pca_train = pca.fit_transform(scaled_train)
pca_test = pca.transform(scaled_test)

# Entrenar el modelo de regresión lineal con características transformadas por PCA
model.fit(pca_train, y_train)
y_pred_pca = model.predict(pca_test)
r2_pca = r2_score(y_test, y_pred_pca)
print("PCA con base en R cuadrada = ", r2_pca)

#Eliminar caracteristicas irrelevantes
selector = SelectKBest(score_func=f_regression, k=5)
selected_train = selector.fit_transform(scaled_train, y_train)
selected_test = selector.transform(scaled_test)

#entrenamiento de modelo de regresion lineal con seleccion de caracteristica
model.fit(selected_train, y_train)
y_predict_selected = model.predict(selected_test)
r2_selected = r2_score(y_test, y_predict_selected)
print("Valor de la seleccion con base en r cuadrada =", r2_selected)

#Transformacion de las varuables
y_train_var = np.log(y_train)
model.fit(scaled_train, y_train_var)
y_pred_var = model.predict(scaled_test)
y_pred_original_scale = np.exp(y_pred_var)
r2_transformada = r2_score(y_test, y_pred_original_scale)
print("R-squared with Log Transformation =", r2_transformada)