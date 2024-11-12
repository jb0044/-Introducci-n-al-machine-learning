#!/usr/bin/env python
# coding: utf-8

# ## Descripción del proyecto
# 
# La compañía móvil Megaline no está satisfecha al ver que muchos de sus clientes utilizan planes heredados. Quieren desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.
# 
# Tienes acceso a los datos de comportamiento de los suscriptores que ya se han cambiado a los planes nuevos (del proyecto del sprint de Análisis estadístico de datos). Para esta tarea de clasificación debes crear un modelo que escoja el plan correcto. Como ya hiciste el paso de procesar los datos, puedes lanzarte directo a crear el modelo.
# 
# Desarrolla un modelo con la mayor exactitud posible. En este proyecto, el umbral de exactitud es 0.75. Usa el dataset para comprobar la exactitud.

# ## Instrucciones del proyecto.
# 
#     - Abre y examina el archivo de datos. Dirección al archivo:datasets/users_behavior.csv Descarga el dataset
#     - Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.
#     - Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.
#     - Comprueba la calidad del modelo usando el conjunto de prueba.
#     - Tarea adicional: haz una prueba de cordura al modelo. Estos datos son más complejos que los que habías usado antes así que no será una tarea fácil. Más adelante lo veremos con más detalle.

# In[1]:


#Importación de librerías

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report



# In[2]:


#Se abre y examina el archivo de datos
df = pd.read_csv('/datasets/users_behavior.csv')

print(df.shape)
print(df.head(5))


# In[3]:


#Segmentación de datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba

features = df.drop(['is_ultra'], axis=1) 
target = df['is_ultra'] 

# Primero dividimos los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Luego, dividimos el conjunto de entrenamiento nuevamente en conjunto de entrenamiento y conjunto de validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# In[4]:


#Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.



for depth in range(1, 6):
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model.fit(X_train, y_train)
    predictions_valid = model.predict(X_val)
    print('max_depth =', depth, ': ', end='')
    print(accuracy_score(y_val, predictions_valid))
    
  


# In[5]:


best_score = 0
best_est = 0
for est in range(1, 11):
    model_final = RandomForestClassifier(random_state=54321, n_estimators=est)
    model_final.fit(X_train, y_train)
    score = model_final.score(X_val, y_val)
    if score > best_score:
        best_score = score
        best_est = est

print("La exactitud del mejor modelo en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))


# In[6]:


df_train, df_valid = train_test_split(df, test_size=0.25, random_state=54321)
93103449


model = LogisticRegression(random_state=54321, solver='liblinear')
model.fit(X_train, y_train)
score_train = model.score(X_train, y_train)
score_valid = model.score(X_val, y_val)

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)


# Una vez que se han investigado los diferentes modelos cambiando los hiperparámetros, se detecta que el RandomForestClassifier tiene el mayor accuracy score, se puede usar ahora este modelo con el set de validación para obtener el nivel de accuracy para esta configuración.
# 

# In[7]:


model_final.fit(X_val, y_val)


# In[9]:


#Comprueba la calidad del modelo usando el conjunto de prueba


results = model_final.score(X_test, y_test)
results_data = model_final.score(X_train, y_train)

print(f"Results with the best fitting model using test set is: {results}")
print(f"Results with the best fitting model using the complete dataset is: {results_data}")#Comprueba la calidad del modelo usando el conjunto de prueba





results = model_final.score(X_test, y_test)

results_data = model_final.score(X_train, y_train)



print(f"Results with the best fitting model using test set is: {results}")

print(f"Results with the best fitting model using the complete dataset is: {results_data}")



# A continuación la prueba de cordura al modelo.

# In[ ]:


# Crear y entrenar el DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# Realizar predicciones con el DummyClassifier
y_pred_dummy = dummy_clf.predict(X_test)

# Evaluar el rendimiento del DummyClassifier
print("Reporte de clasificación del DummyClassifier:")
print(classification_report(y_test, y_pred_dummy))


# La prueba de cordura al modelo demuestra que existe calidad en el modelo, ya que el accuracy del modelo supera al de la prueba de la cordura.

# ## Resumen y conclusiones de proyecto.
# - El objetivo del proyecto es desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.
# - Para esta tarea de clasificación debes crear un modelo que escoja el plan correcto.
# - Se debe desarrollar un modelo con la mayor exactitud posible, en este proyecto el umbral de exactitud es 0.75, se usará el dataset para comprobar la exactitud.
# - Se inicia con la importación de librerías que son necesarias para el desarrollo de este proyecto, que en sí se aplicaron en su mayoría durante los ejercicios de las lecciones del Sprint y pertenecen a sklearn.
# - Se abre y examina el archivo de datos en la dirección al archivo:datasets/users_behavior.csv, se descarga el dataset, y se observan las características generales de la tabla en donde se cuenta con 5 columnas; calls, minutes, messages, mb_used y is_ultra.
# - A continuación se procede con la segmentación de datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba. En donde se definieron el target que es la columna de 'is_ultra' y las features que son el resto de las columnas.
# - Después se dividen los datos en conjunto de entrenamiento y conjunto de prueba, luego dividimos el conjunto de entrenamiento nuevamente en conjunto de entrenamiento y conjunto de validación.
# - Se procede con la investigación de la calidad de diferentes modelos cambiando los hiperparámetros. En donde se prueban los 3 modelos; DecisionTreeClassifier, RandomForestClassifier y LogisticRegression con la finalidad de detectar cual modelo es el de mayor accuracy o exactitud.
# - Una vez que se han investigado los diferentes modelos cambiando los hiperparámetros, se detecta que el RandomForestClassifier tiene el mayor accuracy score, se puede usar ahora este modelo con el set de validación para obtener el nivel de accuracy para esta configuración. Este modelo alcanzó una accuracy del 0.79, la cifra cumple con el objetivo del proyecto de cumplir con una exactitud mayor a 0.75. 
# - Se procede a comprobar la calidad del modelo usando el conjunto de prueba. Los resultados son los siguientes: Resultados con el mejor modelo adecuado usando el set de prueba es: 0.7869362363919129 y resultados con el mejor modelo adecuado usando el dataset completo es: 0.7816390041493776. Estos resultados generados por las pruebas cumplen con la calidad exigida por al proyecto, con un accuracy mayor a 0.75.
# - Y en la parte final del proyecto, se lleva a cabo una prueba de cordura al modelo, en la cual se aplica un DummyClassifier con la estrategia 'most frequent' en donde se crea y se entrena el Dummy, después se realizan predicciones y se evalua el rendimiento del DummyClassifier.
# - El Reporte de clasificación del DummyClassifier arroja un accuracy del 0.71. La prueba de cordura al modelo demuestra que existe calidad en el modelo, ya que el accuracy del modelo supera al de la prueba de la cordura.
# 
# 
# 
