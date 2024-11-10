# %% [markdown]
# # Importar librerías y cargar datos

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %%
df = pd.read_csv('ARREGLO_DIRECTO.csv', delimiter=';')

# %%
df.head(5)

# %%
df.info()

# %%
# Se crea una función que realiza una exploración general a la base de datos
def descripcion(df):
  print("------------------ Descripción general de los datos ------------------")
  print("")
  print("En total la base de datos tiene " + str(len(df)) + " registros y " + str(df.shape[1])+ " variables.")
  print("")
  if df.isnull().any().any():
    print("Algunas columnas tienen valores faltantes. A continuación se muestra el porcentaje de valores nulos por columna:")
    print(df.isnull().sum() / len(df))
  else:
    print("La base de datos no tiene columnas con valores faltantes")

# %%
descripcion(df)

# %% [markdown]
# # Procesamiento de datos

# %%
REGION_DEPTO = {
    'ATLANTICO': 'Caribe',
    'BOLIVAR': 'Caribe',
    'CESAR': 'Caribe',
    'CORDOBA': 'Caribe',
    'SUCRE': 'Caribe',
    'SAN ANDRES': 'Caribe',
    'CAUCA': 'Pacífica',
    'VALLE DEL CAUCA': 'Pacífica',
    'NARIÑO': 'Pacífica',
    'BOGOTA': 'Andina',
    'CUNDINAMARCA': 'Andina',
    'HUILA': 'Andina',
    'TOLIMA': 'Andina',
    'QUINDIO': 'Andina',
    'RISARALDA': 'Andina',
    'SANTANDER': 'Andina',
    'N. DE SANTANDER': 'Andina',
    'META': 'Orinoquia'}




# %%
# Eliminar caracteres no numéricos y convertir a numérico
df['VALOR_PRODUCTO'] = df['VALOR_PRODUCTO'].replace({r'[^\d.]': '', 'INDETERMINADO': None}, regex=True)
df['VALOR_PRODUCTO'] = pd.to_numeric(df['VALOR_PRODUCTO'], errors='coerce')

# Calcular el promedio excluyendo ceros y NaN
promedio = df.loc[df['VALOR_PRODUCTO'] > 0, 'VALOR_PRODUCTO'].mean()

# Imputar los valores de '0' y NaN con el promedio calculado
df['VALOR_PRODUCTO'] = df['VALOR_PRODUCTO'].apply(lambda x: promedio if pd.isna(x) or x == 0 else x)

df['REGION'] = df['UNIDAD_DEPARTAMENTO'].map(REGION_DEPTO)

# %%
variables_interes = ['REGION','ATENCION_TEMA', 'PERSONA_RANGO_EDAD', 'PERSONA_GENERO', 'PERSONA_PROFESION', 'TIPO_PRODUCTO', 'VALOR_PRODUCTO', 'DURACION']
data = df[variables_interes].copy()

# %%
# Codificar variables categóricas
categorical_columns = ['REGION','ATENCION_TEMA', 'PERSONA_RANGO_EDAD', 'PERSONA_GENERO', 'PERSONA_PROFESION', 'TIPO_PRODUCTO']
data_final = pd.get_dummies(data, columns=categorical_columns).astype(int)

# %%
# Identificar las variables independientes
X = data_final.drop('DURACION', axis=1)
# Identificar la variable dependiente
y = data_final['DURACION']

# %%
# Dividir la muestra
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

# %% [markdown]
# ## Modelo 2 con menos variables
# 

# %%
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Filtrar solo las variables de interés
variables_interes = ['REGION', 'ATENCION_TEMA', 'PERSONA_RANGO_EDAD', 'PERSONA_GENERO',
                     'PERSONA_PROFESION', 'TIPO_PRODUCTO', 'VALOR_PRODUCTO', 'DURACION']
data = df[variables_interes].copy()

# Dividir en características (X) y variable objetivo (y)
X = data.drop('DURACION', axis=1)  # Asumimos 'DURACION' como variable objetivo
y = data['DURACION']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear transformador para variables categóricas
categorical_features = ['REGION', 'ATENCION_TEMA', 'PERSONA_RANGO_EDAD', 'PERSONA_GENERO', 'PERSONA_PROFESION', 'TIPO_PRODUCTO']
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Crear preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Pipeline que incluye el preprocesamiento y el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', GradientBoostingRegressor(random_state=42))])

# Definir el espacio de búsqueda para Grid Search
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.8, 1.0]
}

# Configurar Grid Search con validación cruzada
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Mejor estimador
print("Mejores hiperparámetros:", grid_search.best_params_)
model = grid_search.best_estimator_

# Validación cruzada
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print("MAE promedio con validación cruzada:", -scores.mean())

# Entrenar y evaluar el modelo con los mejores hiperparámetros
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)

# Importancia de características (solo si el modelo lo permite)
feature_importance = model.named_steps['model'].feature_importances_
sorted_idx = np.argsort(feature_importance)
features = model.named_steps['preprocessor'].get_feature_names_out()

top_n = 10
sorted_idx = np.argsort(feature_importance)[-top_n:]
top_features = [features[i] for i in sorted_idx]
top_importance = feature_importance[sorted_idx]

# Graficar
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), top_importance, align='center')
plt.yticks(range(len(sorted_idx)), top_features)
plt.title('Top 10 características más importantes')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()


