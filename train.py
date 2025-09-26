import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar el conjunto de datos
try:
    iris = pd.read_csv("data/iris_dataset.csv")
except FileNotFoundError:
    print("Error: El archivo 'data/iris_dataset.csv' no fue encontrado.")

# Dividir el DataFrame en características (X) y etiquetas (y)
X = iris.drop('target', axis=1)
y = iris['target']

# Iniciar un experimento de MLflow
with mlflow.start_run():
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Inicializar y entrenar el modelo
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Realizar predicciones y calcular la precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Guardar el modelo entrenado en un archivo .pkl
    joblib.dump(model, "model.pkl")

    # Registrar el modelo con MLflow
    mlflow.sklearn.log_model(model, "random-forest-model")

    # Registrar parámetros y métricas
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")
