# Modelo de Predicción: `modelo_prediccion_pipeline.pkl`

## Descripción
Este archivo contiene el pipeline final del modelo de predicción de la duración de los arreglos directos. El modelo fue entrenado utilizando Gradient Boosting Regressor y preprocesado con `ColumnTransformer` para manejar variables categóricas y numéricas.

El pipeline incluye:
1. **Preprocesamiento**: Conversión de variables categóricas a representaciones numéricas con `OneHotEncoder`.
2. **Modelo**: Un estimador Gradient Boosting Regressor optimizado con `GridSearchCV` para encontrar los mejores hiperparámetros.

---

## Pasos seguidos

### 1. Exploración inicial
El archivo `Proyecto_1_Despliegue.ipynb` documenta la exploración inicial de los datos:
- Visualización básica (`df.head()` y `df.info()`).
- Identificación de valores faltantes.
- Imputación de valores en la variable `VALOR_PRODUCTO` usando el promedio calculado.

### 2. Procesamiento de datos
- Se estableció un mapeo entre departamentos y regiones (`REGION_DEPTO`).
- Variables categóricas como `REGION`, `ATENCION_TEMA`, y `TIPO_PRODUCTO` fueron codificadas utilizando `OneHotEncoder`.

### 3. Modelos evaluados
Tres modelos principales se exploraron durante el desarrollo:

1. **Random Forest Regressor**:
   - Utilizando `Optuna` para optimización de hiperparámetros.
   - Métricas evaluadas: MAE, RMSE, y feature importance.

2. **Gradient Boosting Regressor**:
   - Búsqueda de hiperparámetros con `GridSearchCV`.
   - Métricas obtenidas:
     - **MAE promedio**: Evaluado con validación cruzada.
     - **Coeficiente de determinación (R²)**: Evaluación en conjunto de prueba.

3. **Red Neuronal con Keras**:
   - Entrenamiento en múltiples épocas con regularización (`Dropout`) para evitar sobreajuste.
   - Pérdidas (MAE y MSE) visualizadas durante las épocas.

### 4. Selección del modelo final
- **Gradient Boosting Regressor** se seleccionó como modelo final debido a su desempeño consistente y facilidad de interpretación.
- Los hiperparámetros óptimos fueron:
  - `n_estimators`: 200
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `subsample`: 0.8

### 5. Entrenamiento del pipeline
En el archivo `Modelo_inicial_Tablero.py`, se construyó un pipeline final con las siguientes etapas:
- **Preprocesamiento**:
  - Transformación de variables categóricas con `OneHotEncoder`.
- **Modelo**:
  - Gradient Boosting Regressor optimizado con `GridSearchCV`.

### 6. Guardado del modelo
El pipeline completo fue guardado utilizando `joblib`:
```python
joblib.dump(model, 'modelo_prediccion_pipeline.pkl')