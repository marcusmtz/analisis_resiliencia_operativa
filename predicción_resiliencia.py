# -*- coding: utf-8 -*-
"""
## Metodología para la Predicción de la Resiliencia Operativa
## Empresas de Telecomunicaciones Peruanas

Pipeline completo en cuatro fases para predecir la resiliencia operativa a partir del
conjunto de datos de reclamos por avería:

1.  **Análisis Exploratorio de Datos (EDA)**:
    *   **Carga de Datos**: Archivo Excel cargado en pandas omitiendo las primeras 3 filas
        de encabezado; columna renombrada de 'N° Reclamos por avería' a
        'Num_Reclamos_por_averia'.
    *   **Inspección inicial**: Estructura, tipos de datos y estadísticas descriptivas.
    *   **Valores faltantes**: Filas con nulos en 'Num_Reclamos_por_averia' eliminadas.
        Las demás columnas no presentaron nulos.
    *   **Outliers (nivel granular)**: Detección visual mediante boxplot IQR. La asimetría
        granular se reporta como referencia; el capping efectivo ocurre DESPUÉS de agregar
        (ver punto 2) para no perder información al sumar.
    *   **Distribución y frecuencias**: Histogramas de reclamos y countplots de variables
        categóricas (empresa, departamento, servicio, materia, tipo).
    *   **Tendencias temporales**: Series mensuales y anuales del total de reclamos para
        identificar estacionalidad y cambios de nivel.

2.  **Limpieza y Preprocesamiento de Datos**:
    *   **Duplicados**: Filas duplicadas identificadas y eliminadas del DataFrame original.
    *   **Agregación**: Registros granulares agrupados por `Mes`, `Empresa operadora`,
        `Departamento`, `Servicio involucrado`, `Materia reclamable` y `Tipo de reclamo`,
        sumando reclamos en `Num_Reclamos_por_averia_Aggregated`. Éste es el nivel
        de granularidad donde operan todos los modelos.
    *   **Capping anti-leakage (winsorización)**: Los límites IQR se estiman SOLO sobre
        el 70% cronológicamente más antiguo (tramo de entrenamiento), y se aplican al
        DataFrame completo. Esto evita que información de val/test contamine los umbrales.
    *   **Características temporales**: `Year`, `Month`, `Quarter` y `Semester` extraídos
        de `Mes`.
    *   **Lag features (1, 3, 6, 12 meses)** y **rolling statistics (ventanas 3, 6, 12)**:
        Calculados por grupo completo `Empresa × Departamento × Servicio × Materia × Tipo`
        con `shift(1)` previo a cada ventana para evitar data leakage. NaN rellenados
        con centinela `-1` ("sin historial"), diferenciado de 0 reclamos reales.
    *   **Alta cardinalidad**: Empresas fuera del top 20 agrupadas en 'Other'.
    *   **Codificación categórica**:
        *   CatBoost y LightGBM usan tipo `category` de pandas (nativamente).
        *   XGBoost y Random Forest usan One-Hot Encoding con alineación de columnas
            train/val/test mediante la función `ohe_and_align()`.
    *   **Variable objetivo `Resiliencia_Operativa`**: Continua, `RO = 1 / (1 + N_reclamos)`
        ∈ (0, 1], donde 1 = máxima resiliencia. La categorización en Baja/Media/Alta
        (percentiles 33/66) se calcula sobre el conjunto de entrenamiento para evitar
        leakage; se usa solo para evaluación descriptiva, no como feature.

3.  **División de Datos y Modelado**:
    *   **División cronológica**: 70% entrenamiento / 15% validación / 15% prueba,
        ordenados temporalmente. El split se hace por posición después de ordenar por
        Year+Month, preservando la causalidad temporal.
    *   **Exclusión del target**: `Num_Reclamos_por_averia_Aggregated`,
        `Resiliencia_Operativa` y `Resiliencia_Operativa_Category` excluidas de X.
    *   **Entrenamiento base** (pre-Optuna) para los 4 modelos con hiperparámetros por
        defecto razonables, usando early stopping sobre el conjunto de validación.
        Métricas base guardadas como snapshots (`*_base`) antes de optimizar.
    *   **Optimización con Optuna (50 trials por modelo)**:
        *   **CatBoost**: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`,
            `bagging_temperature`. El objetivo de Optuna usa validación cruzada temporal
            interna (TimeSeriesSplit, 3 splits sobre train+val) para evitar que los
            hiperparámetros sobreajusten a un único período de validación. Reentrenado
            en train + val con los mejores parámetros encontrados.
        *   **XGBoost**: `eta`, `max_depth`, `subsample`, `colsample_bytree`,
            `min_child_weight`, `gamma`, `n_estimators`. Requiere OHE. Reentrenado
            en train + val.
        *   **LightGBM**: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`,
            `feature_fraction`, `bagging_fraction`, `bagging_freq`, `lambda_l1`,
            `lambda_l2`, `min_child_samples`. Categóricas nativas. Reentrenado en
            train + val (sin early stopping en la fase final).
        *   **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`,
            `min_samples_leaf`, `max_features`. Requiere OHE. Reentrenado en
            train + val.
    *   **Clipping de predicciones negativas**: `np.maximum(0, y_pred)` antes de derivar
        `Resiliencia_Operativa`, ya que reclamos negativos carecen de significado físico.

4.  **Evaluación y Comparación de Modelos**:
    *   **Tablas de métricas con identificación clara**:
        *   [TABLA 1 de 3] Modelos base sin Optuna — referencia de partida.
        *   [TABLA INTERMEDIA] Estado parcial (CatBoost+XGBoost optimizados,
            LightGBM+RF aún base) — solo informativa, no usar para comparar.
        *   [TABLA 3 de 3] Comparación definitiva con los 4 modelos optimizados.
        Las tablas 1 y 3 se exportan automáticamente a TXT (`metricas_modelos_base.txt`
        y `metricas_modelos_optimizados.txt`) para consulta externa.
    *   **Métricas sobre test set**: `MAE`, `RMSE`, `R²` y `MSE` (Error Cuadrático Medio).
        Se prefiere MSE sobre MAPE porque el dataset contiene muchos grupos con 1–3 reclamos
        mensuales, lo que infla artificialmente los errores porcentuales.
    *   **Importancia de características**: Mostrada exactamente dos veces mediante la
        función reutilizable `plot_feature_importance()`: tras los modelos base
        (CatBoost + XGBoost) y tras completar las 4 optimizaciones Optuna (los 4 modelos).
    *   **SHAP (interpretabilidad)**: Beeswarm plots titulados "Impacto y Dirección de
        las Variables" para los 4 modelos. Para Random Forest se submuestrea a 500
        observaciones por coste computacional.
    *   **Validación cruzada temporal (TimeSeriesSplit, 5 folds)**: Evalúa estabilidad
        sobre train + val combinados, usando los mejores hiperparámetros Optuna.
        Reporta distribuciones de RMSE, MAE, R² y MSE por fold.
    *   **Evaluación sobre escala RO (0–1)**: Realizada exclusivamente sobre LightGBM
        (modelo ganador). Incluye MAE y RMSE sobre la escala (0, 1], accuracy de
        categorización en Baja/Media/Alta con umbrales del train set (sin leakage),
        y matriz de confusión individual.

5.  **Pronóstico 3 Meses — LightGBM**:
    *   **Forecasting recursivo**: Para cada mes futuro (T+1, T+2, T+3) se construyen
        las features de lag y rolling a partir del historial disponible. La predicción
        de cada mes alimenta como `lag_1` al mes siguiente, propagando la información
        hacia adelante sin usar datos reales futuros.
    *   **Granularidad**: Se genera una fila por cada combinación única de
        `Empresa × Departamento × Servicio × Materia × Tipo`. La RO reportada es el
        promedio mensual sobre todos los grupos.
    *   **Categorización del pronóstico**: Cada mes pronosticado se clasifica como
        Baja/Media/Alta usando los mismos umbrales p33/p66 del train set.
    *   **Visualización**: Gráfico de línea con la RO real del test set y los 3 meses
        pronosticados, separados por una línea vertical que marca el fin del dataset.

## 1. Exploratory Data Analysis (EDA)

### 1.1 Caracterización del Conjunto de Datos
"""

# ==================================================
# PASO 0: INSTALACIÓN DE DEPENDENCIAS
# ==================================================
# Ejecutar una sola vez para instalar las bibliotecas necesarias:
# pip install catboost optuna lightgbm shap jinja2

# ==================================================
# PASO 1: IMPORTACIONES
# ==================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from catboost import CatBoostRegressor
import xgboost as xgb
from xgboost.callback import EarlyStopping
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import optuna
import shap




def plot_feature_importance(feature_names, importances, title, top_n=10, x_label='Importancia'):
    """Grafica y retorna un DataFrame con la importancia de características ordenada."""
    df_imp = pd.DataFrame({
        'Feature': list(feature_names),
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    print(f"\n{title} (Top {top_n}):")
    print(df_imp)
    fig_h = 7 if top_n <= 10 else 8
    font_sz = 9 if top_n <= 10 else 8
    plt.figure(figsize=(12, fig_h))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(f'{title} (Top {top_n})')
    plt.xlabel(x_label)
    plt.ylabel('Característica')
    for patch in plt.gca().patches:
        plt.gca().annotate(
            f'{patch.get_width():.2f}',
            (patch.get_width(), patch.get_y() + patch.get_height() / 2.),
            ha='left', va='center', fontsize=font_sz, weight='bold'
        )
    plt.tight_layout()
    plt.show()
    return df_imp


def ohe_and_align(X_tr, X_va, X_te, cat_cols):
    """One-Hot Encoding + alineación de columnas entre train, val y test."""
    X_tr_enc = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
    X_va_enc = pd.get_dummies(X_va, columns=cat_cols, drop_first=True)
    X_te_enc = pd.get_dummies(X_te, columns=cat_cols, drop_first=True)
    cols = X_tr_enc.columns
    for c in set(cols) - set(X_va_enc.columns):
        X_va_enc[c] = 0
    for c in set(cols) - set(X_te_enc.columns):
        X_te_enc[c] = 0
    return X_tr_enc, X_va_enc[cols], X_te_enc[cols]


file_path = "D:\\Estudios\\Universidad\\Ciclo 7 2026-1\\Analisis multivariable\\Resiliencia Operativa\\10.3 RECLAMOS POR AVERÍAS.xlsx"
# Reloading the dataset, skipping the first 3 rows and using the 4th row (index 3) as header
df = pd.read_excel(file_path, skiprows=3)

print("Dataset loaded successfully with correct headers.")

print("First 5 rows of the dataset (after reload):")
print(df.head())

print("\nDataset Information (after reload):")
df.info()

print("\nDescriptive Statistics (after reload):")
print(df.describe(include='all'))

# Limpieza de columnas fantasma e indexación
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Ajuste de tipos de datos crítico
# AÑADIDO: Renombramos la columna 'N° Reclamos por avería' a 'Num_Reclamos_por_averia' si existe con el nombre antiguo
if 'N° Reclamos por avería' in df.columns:
    df.rename(columns={'N° Reclamos por avería': 'Num_Reclamos_por_averia'}, inplace=True)

df['Num_Reclamos_por_averia'] = pd.to_numeric(df['Num_Reclamos_por_averia'], errors='coerce')
if 'Mes' in df.columns:
    df['Mes'] = pd.to_datetime(df['Mes'])

# Remove 'DayOfWeek' if it was accidentally created or exists from previous runs
if 'DayOfWeek' in df.columns:
    df.drop(columns=['DayOfWeek'], inplace=True)
    print("Removed 'DayOfWeek' column as it's not relevant for monthly data.")

print("==================================================")
print("     1. CARACTERIZACIÓN DEL CONJUNTO DE DATOS     ")
print("==================================================\n")

n, p = df.shape
print(f"• Número de registros (n): {n:,}")
print(f"• Número de variables (p): {p}")
print("-" * 50)

print("• Tipos de variables por columna:")
print(df.dtypes)
print("-" * 50)

if 'Mes' in df.columns:
    print(f"• Período temporal: {df['Mes'].min().strftime('%Y-%m-%d')} a {df['Mes'].max().strftime('%Y-%m-%d')}")
    print("-" * 50)

print("\n--- ESTADÍSTICOS DESCRIPTIVOS GENERALES ---")
print("\n> Variables Numéricas:")
print(df.describe())
print("\n> Variables Categóricas:")
print(df.describe(include=['object', 'category']))

"""### 1.2 Calidad de los Datos"""

print("==================================================")
print("             1.2 CALIDAD DE LOS DATOS              ")
print("==================================================\n")

# --- 2.1 Valores Faltantes ---
print("--- ANÁLISIS DE VALORES FALTANTES (NULOS) ---")
valores_nulos = df.isnull().sum()
porcentaje_nulos = (df.isnull().sum() / len(df)) * 100
tabla_nulos = pd.DataFrame({'Nulos': valores_nulos, 'Porcentaje (%)': porcentaje_nulos})
print(tabla_nulos[tabla_nulos['Nulos'] > 0] if valores_nulos.sum() > 0 else "¡Excelente! No hay valores faltantes.")
print("-" * 50)

# --- 2.2 Valores Atípicos (Outliers) en Reclamos ---
print("--- ANÁLISIS DE VALORES ATÍPICOS (OUTLIERS) ---")
if 'Num_Reclamos_por_averia' in df.columns:
    # Método IQR (Rango Intercuartílico)
    Q1 = df['Num_Reclamos_por_averia'].quantile(0.25)
    Q3 = df['Num_Reclamos_por_averia'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df['Num_Reclamos_por_averia'] < limite_inferior) | (df['Num_Reclamos_por_averia'] > limite_superior)]

    print(f"• Umbral superior para considerar Outlier: {limite_superior:.2f} reclamos")
    print(f"• Cantidad de registros atípicos detectados: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

    # Gráfico de Caja (Boxplot) para visualizar la dispersión y los outliers
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df['Num_Reclamos_por_averia'], color='salmon')
    plt.title('Detección Visual de Atípicos en Número de Reclamos')
    plt.xlabel('Número de Reclamos por Avería')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
else:
    print("No se encontró la columna numérica para calcular atípicos.")

"""#### 1.2.1 Tratamiento de Valores Atípicos (Outliers)

### 1.3 Análisis Exploratorio de Variables
"""

print("--- TRATAMIENTO DE VALORES ATÍPICOS (CAPPING) ---")
print("El capping (winsorización) se aplica sobre 'Num_Reclamos_por_averia_Aggregated'")
print("tras la agregación, ya que el modelo opera sobre totales agrupados y sumar")
print("valores granulares capeados puede generar igualmente agregados extremos.")
if 'Num_Reclamos_por_averia' in df.columns:
    print(f"• Asimetría granular antes de agregar: {df['Num_Reclamos_por_averia'].skew():.4f}")

print("==================================================")
print("    1.3 ANÁLISIS EXPLORATORIO DE VARIABLES        ")
print("==================================================\n")

# --- 3.1 Distribución de variables categóricas principales ---
plt.figure(figsize=(12, 5))
# Gráfico de las empresas operadoras con más registros
sns.countplot(data=df, y='Empresa operadora', order=df['Empresa operadora'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Empresas Operadoras con Mayor Volumen de Registros')
plt.xlabel('Cantidad de Registros')
plt.ylabel('Empresa')
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=9, weight='bold')
plt.show()

# --- 3.2 Relaciones importantes entre variables ---
print("--- Análisis de Relaciones Clave ---")
# Agrupamos para ver qué empresa acumula la mayor cantidad real de reclamos acumulados
ranking_reclamos = df.groupby('Empresa operadora')['Num_Reclamos_por_averia'].sum().sort_values(ascending=False).head(10)
print("\n> Top 10 Empresas con más Reclamos Totales Acumulados:")
print(ranking_reclamos)

plt.figure(figsize=(12, 5))
ranking_reclamos.plot(kind='bar', color='skyblue')
plt.title('Total de Reclamos Acumulados por Empresa Operadora (Top 10)')
plt.ylabel('Suma Total de Reclamos')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, weight='bold')
plt.tight_layout()
plt.show()

"""### 1.4 Verificación de Valores Faltantes

### Nota importante:

Ahora que hemos categorizado la `Resiliencia_Operativa` para el análisis descriptivo, ten en cuenta que el modelo que entrenamos (CatBoost y XGBoost) predice `Num_Reclamos_por_averia` (un valor numérico) y luego *derivamos* la `Resiliencia_Operativa` continua de esa predicción.

Si quisieras entrenar un modelo para *predecir directamente* esta `Resiliencia_Operativa_Category` (es decir, un problema de clasificación), tendrías que redefinir la variable objetivo de tus modelos y volver a ejecutar los pasos de preparación de datos (`df_encoded`, `X_train`, `X_test`) y el entrenamiento de los modelos para clasificación.

### Checking for Missing Values

### 1.5 Manejo de Valores Faltantes y Duplicados
"""

print("Missing values before handling:")
print(df.isnull().sum())

"""### Handle Missing Values and Duplicates

### 1.6 Análisis de Distribución de Características Clave
"""

# Eliminar filas con valores faltantes en «Num_Reclamos_por_averia»
df.dropna(subset=['Num_Reclamos_por_averia'], inplace=True)

print("Missing values after handling:")
print(df.isnull().sum())

# Buscar filas duplicadas
duplicates_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates_count}")

if duplicates_count > 0:
    print("Dropping duplicate rows...")
    df.drop_duplicates(inplace=True)
    print(f"Number of rows after dropping duplicates: {len(df)}")
else:
    print("No duplicate rows found.")

"""### Distribution Analysis of Key Features

### 1.7 Análisis de Series de Tiempo: Extracción de Características Temporales y Visualización de Tendencias
"""

plt.figure(figsize=(10, 6))
sns.histplot(df['Num_Reclamos_por_averia'], bins=50, kde=True)
plt.title('Distribution of Number of Claims (Num_Reclamos_por_averia)')
plt.xlabel('Number of Claims')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Gráfico filtrado: solo registros con <= 100 reclamos para ver la distribución principal
_df_filtrado = df[df['Num_Reclamos_por_averia'] <= 100]
_pct = len(_df_filtrado) / len(df) * 100
print(f"\n[INFO] Histograma filtrado: {len(_df_filtrado):,} registros ({_pct:.1f}% del total) con Num_Reclamos_por_averia <= 100")

plt.figure(figsize=(10, 6))
sns.histplot(_df_filtrado['Num_Reclamos_por_averia'], bins=50, kde=True, color='steelblue')
plt.title(f'Distribución de Reclamos (filtrado: ≤ 100) — {_pct:.1f}% del total')
plt.xlabel('Num_Reclamos_por_averia')
plt.ylabel('Frecuencia')
plt.xlim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.show()

# Analizar características categóricas
categorical_cols = ['Empresa operadora', 'Departamento', 'Servicio involucrado', 'Materia reclamable', 'Tipo de reclamo']

for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    for p in plt.gca().patches:
        plt.gca().annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=8, weight='bold')
    plt.tight_layout()
    plt.show()

"""### Time-Series Analysis: Extracting Time Features and Visualizing Trends

### 1.8 Preparación para Ingeniería de Características
"""

# Extraer el año y el mes de «Mes»
df['Year'] = df['Mes'].dt.year
df['Month'] = df['Mes'].dt.month

# Agrupa por mes y año para ver las tendencias
monthly_claims = df.groupby(['Year', 'Month'])['Num_Reclamos_por_averia'].sum().reset_index()
monthly_claims['Date'] = pd.to_datetime(monthly_claims['Year'].astype(str) + '-' + monthly_claims['Month'].astype(str))

plt.figure(figsize=(15, 7))
sns.lineplot(data=monthly_claims, x='Date', y='Num_Reclamos_por_averia')
plt.title('Total Number of Claims Over Time (Monthly)')
plt.xlabel('Date')
plt.ylabel('Total Number of Claims')
plt.grid(True)
plt.show()

# Agrupa por año para ver las tendencias anuales
yearly_claims = df.groupby('Year')['Num_Reclamos_por_averia'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_claims, x='Year', y='Num_Reclamos_por_averia', palette='viridis')
plt.title('Total Number of Claims Over Time (Yearly)')
plt.xlabel('Year')
plt.ylabel('Total Number of Claims')
plt.grid(True)
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, weight='bold')
plt.show()

"""### Correlation Analysis (if applicable) and Feature Engineering Prep

## 2. Limpieza y Preprocesamiento de Datos
"""

print("Updated DataFrame head with new time features:")
print(df.head())

print("\nDataFrame Information after adding time features:")
df.info()

"""### 2.1 – 2.4 Revisión del Nivel de Agregación, Variables Temporales, RO y Alta Cardinalidad
(construidos en FASE 4 a continuación)
"""

print("\n### FASE 4: REVISIÓN DEL NIVEL DE AGREGACIÓN & VARIABLES TEMPORALES Y FEATURE ENGINEERING")

# --- 3. Revisión del Nivel de Agregación: Mes + Empresa + Departamento + Servicio + Materia + Tipo ---
# Agregamos los datos por las dimensiones clave y sumamos el número de reclamos
df_agg = df.groupby(['Mes', 'Empresa operadora', 'Departamento', 'Servicio involucrado', 'Materia reclamable', 'Tipo de reclamo'])['Num_Reclamos_por_averia'].sum().reset_index()
df_agg.rename(columns={'Num_Reclamos_por_averia': 'Num_Reclamos_por_averia_Aggregated'}, inplace=True)
print("\nAggregated DataFrame head (Mes + Empresa + Departamento + Servicio + Materia + Tipo):")
print(df_agg.head())

# --- Capping sobre datos agregados (winsorización SIN data leakage) ---
# Los límites IQR se estiman SOLO con el 70% cronológico más antiguo (futuro tramo de
# entrenamiento). Calcularlos sobre todo df_agg filtraría información de validación/prueba
# hacia el recorte del target y hacia las lag/rolling features derivadas de él.
_df_cap_order = df_agg.sort_values('Mes')
_cap_train_n = int(len(_df_cap_order) * 0.7)
_train_target_for_cap = _df_cap_order['Num_Reclamos_por_averia_Aggregated'].iloc[:_cap_train_n]
Q1_agg = _train_target_for_cap.quantile(0.25)
Q3_agg = _train_target_for_cap.quantile(0.75)
IQR_agg = Q3_agg - Q1_agg
lim_inf_agg = Q1_agg - 1.5 * IQR_agg
lim_sup_agg = Q3_agg + 1.5 * IQR_agg
df_agg['Num_Reclamos_por_averia_Aggregated'] = df_agg['Num_Reclamos_por_averia_Aggregated'].clip(lim_inf_agg, lim_sup_agg)
print(f"\n• Capping (límites estimados solo en train 70%): [{lim_inf_agg:.2f}, {lim_sup_agg:.2f}]")
print(f"• Asimetría tras capping agregado: {df_agg['Num_Reclamos_por_averia_Aggregated'].skew():.4f}")

plt.figure(figsize=(10, 4))
sns.boxplot(x=df_agg['Num_Reclamos_por_averia_Aggregated'], color='lightgreen')
plt.title('Distribución de Reclamos Agregados después de Capping')
plt.xlabel('Num_Reclamos_por_averia_Aggregated (Tratado)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 4. Variables Temporales: Year, Month, Quarter, Semester ---
# Extraemos características temporales de la columna 'Mes'
df_agg['Year'] = df_agg['Mes'].dt.year
df_agg['Month'] = df_agg['Mes'].dt.month
df_agg['Quarter'] = df_agg['Mes'].dt.quarter
df_agg['Semester'] = (df_agg['Mes'].dt.month - 1) // 6 + 1
print("\nAggregated DataFrame head with new temporal features:")
print(df_agg.head())

# --- Resiliencia Operativa en datos agregados (pre-split) ---
# NOTA: p33/p66 se calculan aquí sobre todo df_agg para uso exploratorio/visualización.
# Resiliencia_Operativa_Category se EXCLUYE de X (línea 640), por lo que no contamina
# el entrenamiento. La categorización final para métricas usa umbrales del train set.
df_agg['Resiliencia_Operativa'] = 1 / (1 + df_agg['Num_Reclamos_por_averia_Aggregated'])
p33_ro_agg = df_agg['Resiliencia_Operativa'].quantile(0.33)
p66_ro_agg = df_agg['Resiliencia_Operativa'].quantile(0.66)
conditions_agg = [
    df_agg['Resiliencia_Operativa'] <= p33_ro_agg,
    (df_agg['Resiliencia_Operativa'] > p33_ro_agg) & (df_agg['Resiliencia_Operativa'] <= p66_ro_agg),
    df_agg['Resiliencia_Operativa'] > p66_ro_agg
]
choices_agg = ['Baja', 'Media', 'Alta']
df_agg['Resiliencia_Operativa_Category'] = np.select(conditions_agg, choices_agg, default='Baja')
print("\nAggregated DataFrame head with re-calculated Resiliencia_Operativa:")
print(df_agg[['Num_Reclamos_por_averia_Aggregated', 'Resiliencia_Operativa', 'Resiliencia_Operativa_Category']].head())

# --- Feature Engineering: Handling High-Cardinality Categorical Features (Empresa operadora) on Aggregated Data ---
top_n = 20
top_operators_agg = df_agg['Empresa operadora'].value_counts().nlargest(top_n).index
df_agg['Empresa_operadora_grouped'] = df_agg['Empresa operadora'].apply(lambda x: x if x in top_operators_agg else 'Other')
df_agg.drop(columns=['Empresa operadora'], inplace=True) # Drop the original
print("\nAggregated DataFrame head after grouping 'Empresa operadora':")
print(df_agg.head())

# --- FASE 6: CATEGÓRICAS NATIVAS para CatBoost ---
# Para CatBoost, convertimos las columnas categóricas a tipo 'category'
categorical_cols_native = [
    'Departamento',
    'Servicio involucrado',
    'Materia reclamable',
    'Tipo de reclamo',
    'Empresa_operadora_grouped'
]
for col in categorical_cols_native:
    if col in df_agg.columns:
        df_agg[col] = df_agg[col].astype('category')

# Se descarta la columna original 'Mes' ya que hemos extraído características temporales de ella.
df_agg.drop(columns=['Mes'], inplace=True)

# El DataFrame preparado para el modelado (con categoricals nativos) es df_agg
df_prepared = df_agg.copy()

print("\nDataFrame final preparado para el modelado (df_prepared):")
print(df_prepared.head())
df_prepared.info()

"""### 2.6 Ingeniería de Características Temporales Avanzadas (Lags y Rolling Statistics)

## 3. Model Building and Evaluation

### FASE 6: FEATURE ENGINEERING TEMPORAL (Lags y Rolling Statistics) + División Train/Val/Test
"""

print("\nGenerating temporal features (lags and rolling statistics) on df_prepared...")

# Asegúrate de que df_prepared esté ordenado por clave de tiempo y de agrupación antes de crear variables de retraso o variables móviles
# Esto es fundamental para evitar la fuga de datos.
df_prepared_fe = df_prepared.sort_values(
    by=['Empresa_operadora_grouped', 'Departamento', 'Servicio involucrado',
        'Materia reclamable', 'Tipo de reclamo', 'Year', 'Month']
).copy()

# Definir claves de agrupación para la ingeniería de características.
# IMPORTANTE: el nivel de agregación es Mes × Empresa × Departamento × Servicio × Materia × Tipo,
# por lo que la serie temporal ÚNICA está definida por las 5 dimensiones categóricas. Si se
# agrupa solo por Empresa+Departamento+Servicio, cada grupo contiene varias filas del MISMO mes
# (distintas Materia/Tipo) y shift(1) mezcla series contemporáneas en vez de dar el mes anterior.
grouping_keys = ['Empresa_operadora_grouped', 'Departamento', 'Servicio involucrado',
                 'Materia reclamable', 'Tipo de reclamo']

# Generar variables de retraso para «Num_Reclamos_por_averia_Aggregated»
for lag in [1, 3, 6, 12]:
    df_prepared_fe[f'lag_{lag}'] = df_prepared_fe.groupby(grouping_keys)['Num_Reclamos_por_averia_Aggregated'].shift(lag)

# Generar características de media móvil y desviación estándar móvil RESPETANDO grupos.
# Se usa transform para mantener el contexto de grupo durante toda la operación.
for window in [3, 6, 12]:
    df_prepared_fe[f'rolling_mean_{window}'] = (
        df_prepared_fe.groupby(grouping_keys)['Num_Reclamos_por_averia_Aggregated']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    df_prepared_fe[f'rolling_std_{window}'] = (
        df_prepared_fe.groupby(grouping_keys)['Num_Reclamos_por_averia_Aggregated']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
    )

# Rellenar NaN de lag features con -1 (centinela para "sin historial disponible").
# Los modelos de árbol interpretan -1 como categoría separada, evitando confundirlo con 0 reclamos.
lag_rolling_cols = [c for c in df_prepared_fe.columns
                    if c.startswith('lag_') or c.startswith('rolling_')]
df_prepared_fe[lag_rolling_cols] = df_prepared_fe[lag_rolling_cols].fillna(-1)

print("Temporal features generated successfully.")
print("DataFrame head with new temporal features:")
print(df_prepared_fe.head())

print("DataFrame information after adding temporal features:")
df_prepared_fe.info()

# --- Análisis de correlación entre lag y rolling features ---
lag_rolling_fe_cols = [c for c in df_prepared_fe.select_dtypes(include=np.number).columns
                       if c.startswith('lag_') or c.startswith('rolling_')]
plt.figure(figsize=(14, 10))
sns.heatmap(df_prepared_fe[lag_rolling_fe_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlación entre Lag y Rolling Features\n(detectar multicolinealidad)', fontsize=13)
plt.tight_layout()
plt.show()

"""### Re-applying TimeSeriesSplit with new Temporal Features"""

# Vuelve a ordenar los datos por «Año» y «Mes» para garantizar la división temporal, tras el procesamiento de datos
df_sorted_fe = df_prepared_fe.sort_values(by=['Year', 'Month']).reset_index(drop=True)

# Definir características (X) y destino (y) con nuevas características temporales
X = df_sorted_fe.drop(['Num_Reclamos_por_averia_Aggregated', 'Resiliencia_Operativa', 'Resiliencia_Operativa_Category'], axis=1)
y = df_sorted_fe['Num_Reclamos_por_averia_Aggregated']

print(f"Original df_sorted_fe shape: {df_sorted_fe.shape}")
print(f"X shape after dropping target and derived columns: {X.shape}")
print(f"y shape: {y.shape}")

# Determinar los puntos de división para Train, Validation y Test (70%, 15%, 15%)
train_split_point = int(len(X) * 0.7)
val_split_point = int(len(X) * 0.85)

X_train = X.iloc[:train_split_point]
y_train = y.iloc[:train_split_point]

X_val = X.iloc[train_split_point:val_split_point]
y_val = y.iloc[train_split_point:val_split_point]

X_test = X.iloc[val_split_point:]
y_test = y.iloc[val_split_point:]

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_val: {y_val.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

print(f"\nTraining data time range: {df_sorted_fe['Year'].iloc[0]}-{df_sorted_fe['Month'].iloc[0]} to {df_sorted_fe['Year'].iloc[train_split_point-1]}-{df_sorted_fe['Month'].iloc[train_split_point-1]}")
print(f"Validation data time range: {df_sorted_fe['Year'].iloc[train_split_point]}-{df_sorted_fe['Month'].iloc[train_split_point]} to {df_sorted_fe['Year'].iloc[val_split_point-1]}-{df_sorted_fe['Month'].iloc[val_split_point-1]}")
print(f"Test data time range: {df_sorted_fe['Year'].iloc[val_split_point]}-{df_sorted_fe['Month'].iloc[val_split_point]} to {df_sorted_fe['Year'].iloc[-1]}-{df_sorted_fe['Month'].iloc[-1]}")

"""### 3.3 Entrenamiento y Optimización de Modelos

### Re-training CatBoost Regressor with new Temporal Features

#### 3.3.1 Reentrenamiento del Regresor CatBoost con Nuevas Características Temporales
"""

print("\nRe-training CatBoost Regressor with new temporal features...")

# Identificar características categóricas para pasarlas a CatBoost de forma nativa
cat_features_for_catboost = X_train.select_dtypes(include='category').columns.tolist()

print(f"Categorical features for CatBoost: {cat_features_for_catboost}")

# Inicializar el regresor CatBoost con características categóricas
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
    cat_features=cat_features_for_catboost
)

# Entrenar el modelo utilizando el conjunto de validación para la detención temprana
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

print("CatBoost Regressor re-training complete.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_cat = cat_model.predict(X_test)

# Evaluar el modelo
mae_cat = mean_absolute_error(y_test, y_pred_cat)
rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
r2_cat = r2_score(y_test, y_pred_cat)

print(f"\nCatBoost Regressor Performance on Test Set (with temporal features):")
print(f"Mean Absolute Error (MAE): {mae_cat:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_cat:.4f}")
print(f"R-squared (R2): {r2_cat:.4f}")

# Snapshot de métricas BASE (antes de la optimización con Optuna), ya que mae_cat/rmse_cat/...
# se SOBRESCRIBEN más adelante con el modelo optimizado. Se usan en las tablas comparativas base.
mae_cat_base, rmse_cat_base, r2_cat_base = mae_cat, rmse_cat, r2_cat
mse_cat_base = mean_squared_error(y_test, y_pred_cat)

"""#### 3.3.2 Reentrenamiento del Regresor XGBoost con Nuevas Características Temporales

### Re-training XGBoost Regressor with new Temporal Features

#### 3.3.3 Optimización del Regresor CatBoost con Optuna
"""

print("\nRe-training XGBoost Regressor with new temporal features...")

categorical_cols_ohe_xgb = X_train.select_dtypes(include='category').columns.tolist()
X_train_xgb, X_val_xgb, X_test_xgb = ohe_and_align(X_train, X_val, X_test, categorical_cols_ohe_xgb)

# Convertir datos al formato DMatrix
dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
dval = xgb.DMatrix(X_val_xgb, label=y_val)
dtest = xgb.DMatrix(X_test_xgb, label=y_test)

# Definir los parámetros de XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42,
    'nthread': -1
}

# Crear una función de llamada de retorno de EarlyStopping para el conjunto de validación
early_stopping_callback = EarlyStopping(
    rounds=50,
    metric_name='rmse',
    data_name='validation_0'
)

# Entrenar el modelo utilizando el conjunto de validación para la detención temprana
trained_xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation_0')],
    callbacks=[early_stopping_callback],
)

print("XGBoost Regressor re-training complete.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_xgb = trained_xgb_model.predict(dtest)

# Evaluar el modelo
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"\nXGBoost Regressor Performance on Test Set (with temporal features):")
print(f"Mean Absolute Error (MAE): {mae_xgb:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb:.4f}")
print(f"R-squared (R2): {r2_xgb:.4f}")

# Snapshot de métricas BASE de XGBoost (se sobrescriben luego con el modelo optimizado).
mae_xgb_base, rmse_xgb_base, r2_xgb_base = mae_xgb, rmse_xgb, r2_xgb
mse_xgb_base = mean_squared_error(y_test, y_pred_xgb)

"""### Re-calculating Feature Importance with new Temporal Features"""

print("\nRe-calculating Feature Importance for CatBoost and XGBoost with new temporal features...")

cat_feature_importance = plot_feature_importance(
    X_train.columns, cat_model.get_feature_importance(),
    'Importancia - CatBoost (modelos base, características temporales)'
)

xgb_importances_dict = trained_xgb_model.get_score(importance_type='gain')
xgb_feature_importance = plot_feature_importance(
    X_train_xgb.columns,
    X_train_xgb.columns.map(xgb_importances_dict).fillna(0),
    'Importancia - XGBoost (modelo base, características temporales)',
    x_label='Importancia (Ganancia)'
)

"""### Model Training: CatBoost Regressor"""

print("\n### FASE 7: OPTIMIZACIÓN DE MODELOS (CatBoost con Optuna)")

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 10.0, log=True),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0,
        'early_stopping_rounds': 50,
    }

    # CV temporal con 2 splits para evitar overfitting a un único conjunto de validación
    tscv = TimeSeriesSplit(n_splits=3)
    X_tr_full = pd.concat([X_train, X_val])
    y_tr_full = pd.concat([y_train, y_val])

    rmses = []
    for tr_idx, va_idx in tscv.split(X_tr_full):
        X_cv_tr, X_cv_va = X_tr_full.iloc[tr_idx], X_tr_full.iloc[va_idx]
        y_cv_tr, y_cv_va = y_tr_full.iloc[tr_idx], y_tr_full.iloc[va_idx]

        # cat_features como índices posicionales dentro del slice actual
        cat_idx = [X_cv_tr.columns.get_loc(c) for c in cat_features_for_catboost if c in X_cv_tr.columns]

        m = CatBoostRegressor(**params, cat_features=cat_idx)
        m.fit(X_cv_tr, y_cv_tr, eval_set=(X_cv_va, y_cv_va), verbose=0)
        rmses.append(np.sqrt(mean_squared_error(y_cv_va, m.predict(X_cv_va))))

    return np.mean(rmses)

print("Starting Optuna optimization for CatBoost...")
study_catboost = optuna.create_study(direction='minimize', study_name='CatBoost_Optimization')
study_catboost.optimize(objective_catboost, n_trials=50, show_progress_bar=True)

print("Optuna optimization for CatBoost complete.")
print(f"Best trial for CatBoost: {study_catboost.best_trial.value:.4f} RMSE")
print("Best hyperparameters for CatBoost:")
print(study_catboost.best_params)

# Volver a entrenar CatBoost con los mejores hiperparámetros en X_train + X_val para obtener el modelo final
print("Re-training CatBoost with optimized hyperparameters on full train set...")
best_catboost_params = study_catboost.best_params.copy()
best_catboost_params['loss_function'] = 'RMSE'
best_catboost_params['eval_metric'] = 'RMSE'
best_catboost_params['random_seed'] = 42
best_catboost_params['verbose'] = 0
best_catboost_params['cat_features'] = cat_features_for_catboost

cat_model = CatBoostRegressor(**best_catboost_params)
cat_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), verbose=0)

print("CatBoost Regressor training with optimized hyperparameters complete.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_cat = cat_model.predict(X_test)

# Evaluar el modelo
mae_cat = mean_absolute_error(y_test, y_pred_cat)
rmse_cat = np.sqrt(mean_squared_error(y_test, y_pred_cat))
r2_cat = r2_score(y_test, y_pred_cat)

mse_cat = mean_squared_error(y_test, y_pred_cat)

print(f"\nCatBoost Regressor Performance on Test Set (Optimized):")
print(f"Mean Absolute Error (MAE): {mae_cat:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_cat:.4f}")
print(f"R-squared (R2): {r2_cat:.4f}")
print(f"Mean Squared Error (MSE): {mse_cat:.4f}%")

"""#### 3.3.4 Optimización del Regresor XGBoost con Optuna

### Model Training: XGBoost Regressor

#### 3.3.5 Optimización del Regresor LightGBM con Optuna
"""

print("\n### FASE 8: OPTIMIZACIÓN DE MODELOS (XGBoost con Optuna)")

# Regenerar OHE con los mismos parámetros para asegurar consistencia tras Optuna
categorical_cols_ohe_xgb = X_train.select_dtypes(include='category').columns.tolist()
X_train_xgb, X_val_xgb, X_test_xgb = ohe_and_align(X_train, X_val, X_test, categorical_cols_ohe_xgb)

# Convertir datos a formato DMatrix
dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
dval = xgb.DMatrix(X_val_xgb, label=y_val)
dtest = xgb.DMatrix(X_test_xgb, label=y_test)

def objective_xgboost(trial):
    # Hiperparámetros a ajustar para XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'seed': 42,
        'nthread': -1,
    }

    early_stopping_callback = EarlyStopping(
        rounds=50,
        metric_name='rmse',
        data_name='validation_0'
    )

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=trial.suggest_int('n_estimators', 100, 2000),
        evals=[(dval, 'validation_0')],
        callbacks=[early_stopping_callback],
    )

    y_pred_val = model.predict(dval)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    return rmse_val

print("Iniciando optimización de Optuna para XGBoost...")
study_xgboost = optuna.create_study(direction='minimize', study_name='XGBoost_Optimization')
study_xgboost.optimize(objective_xgboost, n_trials=50, show_progress_bar=True)

print("Optimización de Optuna para XGBoost completada.")
print(f"Mejor prueba para XGBoost: {study_xgboost.best_trial.value:.4f} RMSE")
print("Mejores hiperparámetros para XGBoost:")
print(study_xgboost.best_params)

# Volver a entrenar XGBoost con los mejores hiperparámetros en X_train + X_val para el modelo final
print("Reentrenando XGBoost con hiperparámetros optimizados en el conjunto de entrenamiento completo...")
best_xgb_params = study_xgboost.best_params.copy()
best_xgb_params['objective'] = 'reg:squarederror'
best_xgb_params['eval_metric'] = 'rmse'
best_xgb_params['seed'] = 42
best_xgb_params['nthread'] = -1
num_boost_round_optimized = best_xgb_params.pop('n_estimators') # Extraer n_estimators

# Combinar conjuntos de entrenamiento y validación para el entrenamiento final
X_train_val_xgb = pd.concat([X_train_xgb, X_val_xgb])
y_train_val = pd.concat([y_train, y_val])
dtrain_val = xgb.DMatrix(X_train_val_xgb, label=y_train_val)

trained_xgb_model = xgb.train(
    best_xgb_params,
    dtrain_val,
    num_boost_round=num_boost_round_optimized,
)

print("Entrenamiento del regresor XGBoost con hiperparámetros optimizados completado.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_xgb = trained_xgb_model.predict(dtest)

# Evaluar el modelo
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)

print(f"\nXGBoost Regressor Performance en el conjunto de prueba (Optimizado):")
print(f"Error Absoluto Medio (MAE): {mae_xgb:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_xgb:.4f}")
print(f"R-squared (R2): {r2_xgb:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse_xgb:.4f}%")

"""### Model Training: LightGBM Regressor

#### 3.3.6 Optimización del Regresor Random Forest con Optuna
"""

print("\n### FASE 9: ENTRENAMIENTO DE MODELOS (LightGBM)")
print("Entrenando el Regresor LightGBM...")

# LightGBM puede manejar características categóricas de forma nativa si se especifica.
# Espera que las características categóricas estén codificadas como enteros o como tipo de datos 'category'.
# X_train ya tiene las características categóricas como tipo de datos 'category'.

# Identificar características categóricas para LightGBM
cat_features_for_lgbm = X_train.select_dtypes(include='category').columns.tolist()

# Convertir características categóricas a tipo de datos 'category' para LightGBM, si aún no lo están.
# (Este paso es técnicamente redundante si `df_prepared` ya las estableció, pero es bueno para la robustez)
for col in cat_features_for_lgbm:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Inicializar el Regresor LightGBM
lgbm_model = lgb.LGBMRegressor(
    objective='regression_l1',
    metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=42,
    n_jobs=-1,
)

# Entrenar el modelo con parada temprana
lgbm_model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                categorical_feature=cat_features_for_lgbm)

print("Entrenamiento del Regresor LightGBM completado.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_lgbm = lgbm_model.predict(X_test)

# Evaluar el modelo
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)

print(f"\nLightGBM Regressor Performance en el conjunto de prueba (con características temporales):")
print(f"Error Absoluto Medio (MAE): {mae_lgbm:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_lgbm:.4f}")
print(f"R-squared (R2): {r2_lgbm:.4f}")
print(f"Mean Squared Error (MSE): {mse_lgbm:.4f}%")

"""### Model Training: Random Forest Regressor

### 4.1 Comparación del Rendimiento del Modelo (Resumen Inicial)
"""

print("\n### FASE 10: ENTRENAMIENTO DE MODELOS (Random Forest)")
print("Entrenando el Regresor Random Forest...")

# RF comparte el OHE generado para XGBoost (mismo espacio de características)
X_train_rf_final = X_train_xgb
X_val_rf_final   = X_val_xgb
X_test_rf_final  = X_test_xgb


# Inicializar el Regresor Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100, # Número de árboles en el bosque
    max_depth=10, # Profundidad máxima de los árboles individuales
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7, # Fracción de características a considerar al buscar la mejor división
    random_state=42,
    n_jobs=-1 # Usar todos los núcleos disponibles
)

# Entrenar el modelo
rf_model.fit(X_train_rf_final, y_train)

print("Entrenamiento del Regresor Random Forest completado.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_rf = rf_model.predict(X_test_rf_final)

# Evaluar el modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f"\nRandom Forest Regressor Performance en el conjunto de prueba (con características temporales):")
print(f"Error Absoluto Medio (MAE): {mae_rf:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_rf:.4f}")
print(f"R-squared (R2): {r2_rf:.4f}")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}%")

"""## 4. Model Comparison and Feature Importance Analysis

### Comparison of Model Performance

### 4.2 Análisis de Importancia de Características (Inicial)
"""

# Crear un diccionario para almacenar el rendimiento del modelo (MODELOS BASE, sin Optuna).
# Se usan los snapshots *_base de CatBoost/XGBoost porque mae_cat/mae_xgb ya fueron
# sobrescritos por sus versiones optimizadas; LightGBM y Random Forest aún son base aquí.
performance_data = {
    'Model': [
        'CatBoost Regressor',
        'XGBoost Regressor',
        'LightGBM Regressor',
        'Random Forest Regressor'
    ],
    'MAE': [mae_cat_base, mae_xgb_base, mae_lgbm, mae_rf],
    'RMSE': [rmse_cat_base, rmse_xgb_base, rmse_lgbm, rmse_rf],
    'R-squared': [r2_cat_base, r2_xgb_base, r2_lgbm, r2_rf],
    'MSE': [mse_cat_base, mse_xgb_base, mse_lgbm, mse_rf]
}

# Crear un DataFrame para facilitar la comparación
performance_df = pd.DataFrame(performance_data)

print("\n" + "="*60)
print("  [TABLA 1 de 3] MODELOS BASE — sin optimización Optuna")
print("  Referencia de partida. Úsala para ver cuánto mejora Optuna.")
print("="*60)
print(performance_df.round(4))
with open('metricas_modelos_base.txt', 'w', encoding='utf-8') as _f:
    _f.write("[TABLA 1 de 3] MODELOS BASE — sin optimización Optuna\n")
    _f.write("Referencia de partida. Úsala para ver cuánto mejora Optuna.\n")
    _f.write("="*60 + "\n")
    _f.write(performance_df.round(4).to_string(index=False))
    _f.write("\n")
print("  >> Guardado en: metricas_modelos_base.txt")

# Resaltar el modelo de mejor rendimiento para cada métrica
def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: lightgreen' if v else '' for v in is_min]

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

styled_performance_df = performance_df.style \
    .apply(highlight_min, subset=['MAE', 'RMSE', 'MSE']) \
    .apply(highlight_max, subset=['R-squared'])

print("\nComparación del Rendimiento del Modelo (con resaltados para el mejor rendimiento):")
print(styled_performance_df)

# ==================================================
# B. EVALUACIÓN DEL RENDIMIENTO DE LOS MODELOS (BASE — sin Optuna)
# ==================================================

# Todos los modelos en su versión BASE (pre-Optuna).
# CatBoost y XGBoost usan snapshots *_base; LightGBM y RF aún no han sido optimizados.
datos_rendimiento = {
    'Modelo': ['CatBoost Regressor', 'XGBoost Regressor', 'LightGBM Regressor', 'Random Forest Regressor'],
    'MAE':  [mae_cat_base,  mae_xgb_base,  mae_lgbm,  mae_rf],
    'RMSE': [rmse_cat_base, rmse_xgb_base, rmse_lgbm, rmse_rf],
    'R²':   [r2_cat_base,   r2_xgb_base,   r2_lgbm,   r2_rf],
    'MSE': [mse_cat_base, mse_xgb_base, mse_lgbm, mse_rf]
}

# 2. Convertimos a DataFrame de Pandas
df_metricas = pd.DataFrame(datos_rendimiento)

print("="*60)
print("  [TABLA 2 de 3] MODELOS BASE con gráfico — sin Optuna")
print("  Mismo contenido que Tabla 1, con visualización de barras.")
print("  MAE/RMSE/MSE: menor es mejor | R²: mayor es mejor.")
print("="*60)
print(df_metricas.to_string(index=False))
print("-" * 50)

# 3. Gráfico Académico Comparativo (Métricas clave: R², RMSE, MAE, MSE)
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('Evaluación Comparativa: Modelos Base', fontsize=14, weight='bold', y=1.02)

# Gráfico A: Coeficiente de Determinación R² (Mayor es mejor)
sns.barplot(data=df_metricas, x='Modelo', y='R²', ax=axes[0], palette='Blues_r', edgecolor='black', width=0.4)
axes[0].set_title('Comparación de la Métrica R²\n(Mayor es mejor)', fontsize=12, pad=10)
axes[0].set_ylabel('Coeficiente R²')
axes[0].set_xlabel('Modelos Evaluados')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', weight='bold', fontsize=10)

# Gráfico B: Raíz del Error Cuadrático Medio RMSE (Menor es mejor)
sns.barplot(data=df_metricas, x='Modelo', y='RMSE', ax=axes[1], palette='Oranges', edgecolor='black', width=0.4)
axes[1].set_title('Comparación del Error RMSE\n(Menor es mejor)', fontsize=12, pad=10)
axes[1].set_ylabel('Valor del RMSE')
axes[1].set_xlabel('Modelos Evaluados')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + 20),
                ha='center', va='center', weight='bold', fontsize=10)

# Gráfico C: Error Absoluto Medio MAE (Menor es mejor)
sns.barplot(data=df_metricas, x='Modelo', y='MAE', ax=axes[2], palette='Greens', edgecolor='black', width=0.4)
axes[2].set_title('Comparación del Error MAE\n(Menor es mejor)', fontsize=12, pad=10)
axes[2].set_ylabel('Valor del MAE')
axes[2].set_xlabel('Modelos Evaluados')
axes[2].grid(axis='y', linestyle='--', alpha=0.5)
for p in axes[2].patches:
    axes[2].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + 20),
                ha='center', va='center', weight='bold', fontsize=10)

# Gráfico D: Error Cuadrático Medio MSE (Menor es mejor)
sns.barplot(data=df_metricas, x='Modelo', y='MSE', ax=axes[3], palette='Reds', edgecolor='black', width=0.4)
axes[3].set_title('Comparación del Error MSE\n(Menor es mejor)', fontsize=12, pad=10)
axes[3].set_ylabel('Valor del MSE')
axes[3].set_xlabel('Modelos Evaluados')
axes[3].grid(axis='y', linestyle='--', alpha=0.5)
for p in axes[3].patches:
    axes[3].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 20),
                ha='center', va='center', weight='bold', fontsize=10)

plt.tight_layout()
plt.show()

"""### Feature Importance Analysis

### 4.3 Comparación del Rendimiento del Modelo (Actualizado)
"""


"""### FASE 8: EVALUACIÓN CIENTÍFICA (Comparación de Modelos Actualizada)

### 4.4 Análisis de Importancia de Características (Actualizado)
"""

# Tabla intermedia: CatBoost y XGBoost ya optimizados; LightGBM y RF todavía son base
performance_data = {
    'Model': [
        'CatBoost (Optimizado)',
        'XGBoost (Optimizado)',
        'LightGBM (BASE — aún sin Optuna)',
        'Random Forest (BASE — aún sin Optuna)'
    ],
    'MAE':       [mae_cat,  mae_xgb,  mae_lgbm,  mae_rf],
    'RMSE':      [rmse_cat, rmse_xgb, rmse_lgbm, rmse_rf],
    'R-squared': [r2_cat,   r2_xgb,   r2_lgbm,   r2_rf],
    'MSE':      [mse_cat, mse_xgb, mse_lgbm, mse_rf]
}

performance_df = pd.DataFrame(performance_data)

print("\n" + "="*60)
print("  [TABLA INTERMEDIA] Estado parcial de optimización")
print("  CatBoost y XGBoost: ya con Optuna.")
print("  LightGBM y RF: todavía son modelos base (Optuna aún no corrió).")
print("  ⚠ NO usar esta tabla para comparar modelos entre sí.")
print("="*60)
print(performance_df.round(4))
with open('metricas_intermedia.txt', 'w', encoding='utf-8') as _f:
    _f.write("[TABLA INTERMEDIA] Estado parcial de optimización\n")
    _f.write("CatBoost y XGBoost: ya con Optuna. LightGBM y RF: todavía base.\n")
    _f.write("⚠ NO usar para comparar modelos entre sí.\n")
    _f.write("="*60 + "\n")
    _f.write(performance_df.round(4).to_string(index=False))
    _f.write("\n")
print("  >> Guardado en: metricas_intermedia.txt")

styled_performance_df = performance_df.style \
    .apply(highlight_min, subset=['MAE', 'RMSE', 'MSE']) \
    .apply(highlight_max, subset=['R-squared'])

print("\n(con resaltados de mejor por métrica):")
print(styled_performance_df)

# --- Visual Comparison of Metrics ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparative Evaluation of Regression Models (R², RMSE, MAE)', fontsize=16, weight='bold', y=1.02)

# R² Score (Higher is better)
sns.barplot(ax=axes[0], x='R-squared', y='Model', data=performance_df.sort_values(by='R-squared', ascending=False), palette='viridis')
axes[0].set_title('R-squared (Higher is Better)')
axes[0].set_xlabel('R-squared Score')
axes[0].set_ylabel('')
axes[0].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_width():.4f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

# RMSE (Lower is better)
sns.barplot(ax=axes[1], x='RMSE', y='Model', data=performance_df.sort_values(by='RMSE', ascending=True), palette='magma')
axes[1].set_title('RMSE (Lower is Better)')
axes[1].set_xlabel('RMSE')
axes[1].set_ylabel('')
axes[1].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

# MAE (Lower is better)
sns.barplot(ax=axes[2], x='MAE', y='Model', data=performance_df.sort_values(by='MAE', ascending=True), palette='cividis')
axes[2].set_title('MAE (Lower is Better)')
axes[2].set_xlabel('MAE')
axes[2].set_ylabel('')
axes[2].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[2].patches:
    axes[2].annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

plt.tight_layout()
plt.show()

"""### FASE 9: INTERPRETABILIDAD (Feature Importance - Updated)

### 4.5 Valores SHAP para la Interpretabilidad del Modelo
"""

print("\n### FASE 11: OPTIMIZACIÓN DE MODELOS (LightGBM con Optuna)")

cat_features_for_lgbm = X_train.select_dtypes(include='category').columns.tolist()

def objective_lgbm(trial):
    # Hiperparámetros a ajustar para LightGBM
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                categorical_feature=cat_features_for_lgbm)

    y_pred_val = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    return rmse_val

print("Iniciando optimización de Optuna para LightGBM...")
study_lgbm = optuna.create_study(direction='minimize', study_name='LightGBM_Optimization')
study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)

print("Optimización de Optuna para LightGBM completada.")
print(f"Mejor prueba para LightGBM: {study_lgbm.best_trial.value:.4f} RMSE")
print("Mejores hiperparámetros para LightGBM:")
print(study_lgbm.best_params)

# Volver a entrenar LightGBM con los mejores hiperparámetros en X_train + X_val para el modelo final
print("Reentrenando LightGBM con hiperparámetros optimizados en el conjunto de entrenamiento completo...")
best_lgbm_params = study_lgbm.best_params.copy()
best_lgbm_params['objective'] = 'regression_l1'
best_lgbm_params['metric'] = 'rmse'
best_lgbm_params['random_state'] = 42
best_lgbm_params['n_jobs'] = -1

lgbm_model = lgb.LGBMRegressor(**best_lgbm_params)

# Combinar conjuntos de entrenamiento y validación para el entrenamiento final
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

lgbm_model.fit(X_train_val, y_train_val,
               categorical_feature=cat_features_for_lgbm)

print("Entrenamiento del Regresor LightGBM con hiperparámetros optimizados completado.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_lgbm = lgbm_model.predict(X_test)

# Evaluar el modelo
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)

print(f"\nLightGBM Regressor Performance en el conjunto de prueba (Optimizado):")
print(f"Error Absoluto Medio (MAE): {mae_lgbm:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_lgbm:.4f}")
print(f"R-squared (R2): {r2_lgbm:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse_lgbm:.4f}%")

print("\n### FASE 12: OPTIMIZACIÓN DE MODELOS (Random Forest con Optuna)")

# RF comparte el mismo OHE que XGBoost (ya calculado en FASE 8)
categorical_cols_ohe_rf = X_train.select_dtypes(include='category').columns.tolist()
X_train_rf_final, X_val_rf_final, X_test_rf_final = ohe_and_align(
    X_train, X_val, X_test, categorical_cols_ohe_rf
)

def objective_rf(trial):
    # Hiperparámetros a ajustar para Random Forest
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train_rf_final, y_train)

    y_pred_val = model.predict(X_val_rf_final)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    return rmse_val

print("Iniciando optimización de Optuna para Random Forest...")
study_rf = optuna.create_study(direction='minimize', study_name='RandomForest_Optimization')
study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=True)

print("Optimización de Optuna para Random Forest completada.")
print(f"Mejor prueba para Random Forest: {study_rf.best_trial.value:.4f} RMSE")
print("Mejores hiperparámetros para Random Forest:")
print(study_rf.best_params)

# Volver a entrenar Random Forest con los mejores hiperparámetros en X_train + X_val para el modelo final
print("Reentrenando Random Forest con hiperparámetros optimizados en el conjunto de entrenamiento completo...")
best_rf_params = study_rf.best_params.copy()
best_rf_params['random_state'] = 42
best_rf_params['n_jobs'] = -1

rf_model = RandomForestRegressor(**best_rf_params)

# Combinar conjuntos de entrenamiento y validación para el entrenamiento final
X_train_val_rf = pd.concat([X_train_rf_final, X_val_rf_final])
y_train_val = pd.concat([y_train, y_val])

rf_model.fit(X_train_val_rf, y_train_val)

print("Entrenamiento del Regresor Random Forest con hiperparámetros optimizados completado.")

# Realizar predicciones sobre el conjunto de prueba
y_pred_rf = rf_model.predict(X_test_rf_final)

# Evaluar el modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f"\nRandom Forest Regressor Performance en el conjunto de prueba (Optimizado):")
print(f"Error Absoluto Medio (MAE): {mae_rf:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_rf:.4f}")
print(f"R-squared (R2): {r2_rf:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse_rf:.4f}%")

# --- Importancia de características: los 4 modelos YA optimizados ---
print("\n" + "="*60)
print("  IMPORTANCIA DE CARACTERÍSTICAS — 4 MODELOS OPTIMIZADOS")
print("="*60)

cat_feature_importance = plot_feature_importance(
    X_train.columns, cat_model.get_feature_importance(),
    'Importancia - CatBoost (optimizado con Optuna)', top_n=20
)

xgb_importances_dict = trained_xgb_model.get_score(importance_type='gain')
xgb_feature_importance = plot_feature_importance(
    X_train_xgb.columns,
    X_train_xgb.columns.map(xgb_importances_dict).fillna(0),
    'Importancia - XGBoost (optimizado con Optuna)', top_n=20, x_label='Importancia (Ganancia)'
)

lgbm_feature_importance = plot_feature_importance(
    X_train.columns, lgbm_model.feature_importances_,
    'Importancia - LightGBM (optimizado con Optuna)', top_n=20
)

rf_feature_importance = plot_feature_importance(
    X_train_rf_final.columns, rf_model.feature_importances_,
    'Importancia - Random Forest (optimizado con Optuna)', top_n=20
)

"""FASE 8: EVALUACIÓN CIENTÍFICA (Tabla Comparativa y Ranking Final de Modelos"""

print("\n### FASE 13: EVALUACIÓN CIENTÍFICA (Tabla Comparativa y Ranking Final de Modelos)\n")

performance_data = {
    'Model': [
        'CatBoost Regressor (Optimizado)',
        'XGBoost Regressor (Optimizado)',
        'LightGBM Regressor (Optimizado)',
        'Random Forest Regressor (Optimizado)'
    ],
    'MAE':       [mae_cat,  mae_xgb,  mae_lgbm,  mae_rf],
    'RMSE':      [rmse_cat, rmse_xgb, rmse_lgbm, rmse_rf],
    'R-squared': [r2_cat,   r2_xgb,   r2_lgbm,   r2_rf],
    'MSE':      [mse_cat, mse_xgb, mse_lgbm, mse_rf]
}

performance_df = pd.DataFrame(performance_data)

print("\n" + "="*60)
print("  [TABLA 3 de 3] COMPARACIÓN DEFINITIVA — 4 MODELOS OPTIMIZADOS CON OPTUNA")
print("  ESTA es la tabla principal para interpretar resultados finales.")
print("  MAE / RMSE / MSE: menor es mejor  |  R²: mayor es mejor.")
print("  Compara con [TABLA 1 de 3] para ver la ganancia de Optuna.")
print("="*60)
print(performance_df.round(4))
with open('metricas_modelos_optimizados.txt', 'w', encoding='utf-8') as _f:
    _f.write("[TABLA 3 de 3] COMPARACIÓN DEFINITIVA — 4 MODELOS OPTIMIZADOS CON OPTUNA\n")
    _f.write("ESTA es la tabla principal para interpretar resultados finales.\n")
    _f.write("MAE / RMSE / MSE: menor es mejor  |  R²: mayor es mejor.\n")
    _f.write("Compara con metricas_modelos_base.txt para ver la ganancia de Optuna.\n")
    _f.write("="*60 + "\n")
    _f.write(performance_df.round(4).to_string(index=False))
    _f.write("\n")
print("  >> Guardado en: metricas_modelos_optimizados.txt")

styled_performance_df = performance_df.style \
    .apply(highlight_min, subset=['MAE', 'RMSE', 'MSE']) \
    .apply(highlight_max, subset=['R-squared'])

print("\n(con resaltados de mejor por métrica):")
print(styled_performance_df)

# --- Comparación Visual de Métricas ---
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle('Evaluación Comparativa de Modelos de Regresión (Optimizado)', fontsize=16, weight='bold', y=1.02)

# Puntuación R² (mayor es mejor)
sns.barplot(ax=axes[0], x='R-squared', y='Model', data=performance_df.sort_values(by='R-squared', ascending=False), palette='Blues_r')
axes[0].set_title('R-squared (Mayor es Mejor)')
axes[0].set_xlabel('Puntuación R-squared')
axes[0].set_ylabel('')
axes[0].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_width():.4f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

# RMSE (menor es mejor)
sns.barplot(ax=axes[1], x='RMSE', y='Model', data=performance_df.sort_values(by='RMSE', ascending=True), palette='Oranges_r')
axes[1].set_title('RMSE (Menor es Mejor)')
axes[1].set_xlabel('RMSE')
axes[1].set_ylabel('')
axes[1].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

# MAE (menor es mejor)
sns.barplot(ax=axes[2], x='MAE', y='Model', data=performance_df.sort_values(by='MAE', ascending=True), palette='Greens_r')
axes[2].set_title('MAE (Menor es Mejor)')
axes[2].set_xlabel('MAE')
axes[2].set_ylabel('')
axes[2].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[2].patches:
    axes[2].annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

# MSE (menor es mejor)
sns.barplot(ax=axes[3], x='MSE', y='Model', data=performance_df.sort_values(by='MSE', ascending=True), palette='Reds_r')
axes[3].set_title('MSE (Menor es Mejor)')
axes[3].set_xlabel('MSE')
axes[3].set_ylabel('')
axes[3].grid(axis='x', linestyle='--', alpha=0.7)
for p in axes[3].patches:
    axes[3].annotate(f'{p.get_width():.2f}%', (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=8, weight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""FASE 14: INTERPRETABILIDAD (Valores SHAP)"""

# --- Valores SHAP de CatBoost ---
print("\nCalculando SHAP para CatBoost...")
explainer_cat = shap.TreeExplainer(cat_model)
shap_exp_cat = explainer_cat(X_test)
_sv = np.array(shap_exp_cat.values)
print(f"  [DEBUG] shap shape: {_sv.shape}, max|SHAP|: {np.abs(_sv).max():.6f} — {'OK' if np.abs(_sv).max() > 0 else 'ADVERTENCIA: todos cero'}")
_ma = np.abs(_sv).mean(axis=0); _t5 = np.argsort(_ma)[::-1][:5]
print("  [DEBUG] Top 5 features (CatBoost): " + ", ".join(f"{list(X_test.columns)[i]}={_ma[i]:.4f}" for i in _t5))

plt.close('all')
shap.plots.beeswarm(shap_exp_cat, max_display=20, show=False)
_shap_fig = plt.gcf()
_shap_fig.set_size_inches(12, 8)
_shap_fig.suptitle('Impacto y Dirección de las Variables — CatBoost', y=1.01, fontsize=13)
_dbg_nc = sum(len(ax.collections) for ax in _shap_fig.axes)
print(f"  [DEBUG] Figura CatBoost — ejes: {len(_shap_fig.axes)}, colecciones: {_dbg_nc} ({'TIENE DATOS' if _dbg_nc > 0 else 'SIN PUNTOS VISIBLES'})")
_fpath = os.path.abspath('shap_catboost.png')
_shap_fig.savefig(_fpath, dpi=150, bbox_inches='tight')
print(f"  [DEBUG] Guardado: {_fpath} ({os.path.getsize(_fpath):,} bytes)")
plt.close('all')
try:
    os.startfile(_fpath)
except AttributeError:
    pass  # os.startfile solo existe en Windows

# --- Valores SHAP de XGBoost ---
print("\nCalculando SHAP para XGBoost...")
explainer_xgb = shap.TreeExplainer(trained_xgb_model)
shap_exp_xgb = explainer_xgb(X_test_xgb)
_sv = np.array(shap_exp_xgb.values)
print(f"  [DEBUG] shap shape: {_sv.shape}, max|SHAP|: {np.abs(_sv).max():.6f} — {'OK' if np.abs(_sv).max() > 0 else 'ADVERTENCIA: todos cero'}")
_ma = np.abs(_sv).mean(axis=0); _t5 = np.argsort(_ma)[::-1][:5]
print("  [DEBUG] Top 5 features (XGBoost): " + ", ".join(f"{list(X_test_xgb.columns)[i]}={_ma[i]:.4f}" for i in _t5))

plt.close('all')
shap.plots.beeswarm(shap_exp_xgb, max_display=20, show=False)
_shap_fig = plt.gcf()
_shap_fig.set_size_inches(12, 8)
_shap_fig.suptitle('Impacto y Dirección de las Variables — XGBoost', y=1.01, fontsize=13)
_dbg_nc = sum(len(ax.collections) for ax in _shap_fig.axes)
print(f"  [DEBUG] Figura XGBoost — ejes: {len(_shap_fig.axes)}, colecciones: {_dbg_nc} ({'TIENE DATOS' if _dbg_nc > 0 else 'SIN PUNTOS VISIBLES'})")
_fpath = os.path.abspath('shap_xgboost.png')
_shap_fig.savefig(_fpath, dpi=150, bbox_inches='tight')
print(f"  [DEBUG] Guardado: {_fpath} ({os.path.getsize(_fpath):,} bytes)")
plt.close('all')
try:
    os.startfile(_fpath)
except AttributeError:
    pass  # os.startfile solo existe en Windows

# --- Valores SHAP de LightGBM ---
print("\nCalculando SHAP para LightGBM...")
explainer_lgbm = shap.TreeExplainer(lgbm_model)
shap_exp_lgbm = explainer_lgbm(X_test)
_sv = np.array(shap_exp_lgbm.values)
print(f"  [DEBUG] shap shape: {_sv.shape}, max|SHAP|: {np.abs(_sv).max():.6f} — {'OK' if np.abs(_sv).max() > 0 else 'ADVERTENCIA: todos cero'}")
_ma = np.abs(_sv).mean(axis=0); _t5 = np.argsort(_ma)[::-1][:5]
print("  [DEBUG] Top 5 features (LightGBM): " + ", ".join(f"{list(X_test.columns)[i]}={_ma[i]:.4f}" for i in _t5))

plt.close('all')
shap.plots.beeswarm(shap_exp_lgbm, max_display=20, show=False)
_shap_fig = plt.gcf()
_shap_fig.set_size_inches(12, 8)
_shap_fig.suptitle('Impacto y Dirección de las Variables — LightGBM', y=1.01, fontsize=13)
_dbg_nc = sum(len(ax.collections) for ax in _shap_fig.axes)
print(f"  [DEBUG] Figura LightGBM — ejes: {len(_shap_fig.axes)}, colecciones: {_dbg_nc} ({'TIENE DATOS' if _dbg_nc > 0 else 'SIN PUNTOS VISIBLES'})")
_fpath = os.path.abspath('shap_lgbm.png')
_shap_fig.savefig(_fpath, dpi=150, bbox_inches='tight')
print(f"  [DEBUG] Guardado: {_fpath} ({os.path.getsize(_fpath):,} bytes)")
plt.close('all')
try:
    os.startfile(_fpath)
except AttributeError:
    pass  # os.startfile solo existe en Windows

# --- Valores SHAP de Random Forest ---
# Se subsamplea X_test_rf_final a 500 filas para acelerar el cálculo SHAP.
# Con 500 muestras el resultado es representativo; usar todo el test set es 10x–100x más lento
# debido a la combinación de hasta 500 árboles, max_depth=20 y cientos de features OHE.
print("\nCalculando SHAP para Random Forest...")
shap_sample_size = min(500, len(X_test_rf_final))
X_shap_rf = X_test_rf_final.sample(n=shap_sample_size, random_state=42)
explainer_rf = shap.TreeExplainer(rf_model)
shap_exp_rf = explainer_rf(X_shap_rf)
_sv = np.array(shap_exp_rf.values)
print(f"  [DEBUG] shap shape: {_sv.shape}, n_muestras: {shap_sample_size}, max|SHAP|: {np.abs(_sv).max():.6f} — {'OK' if np.abs(_sv).max() > 0 else 'ADVERTENCIA: todos cero'}")
_ma = np.abs(_sv).mean(axis=0); _t5 = np.argsort(_ma)[::-1][:5]
print("  [DEBUG] Top 5 features (RF): " + ", ".join(f"{list(X_shap_rf.columns)[i]}={_ma[i]:.4f}" for i in _t5))

plt.close('all')
shap.plots.beeswarm(shap_exp_rf, max_display=20, show=False)
_shap_fig = plt.gcf()
_shap_fig.set_size_inches(12, 8)
_shap_fig.suptitle('Impacto y Dirección de las Variables — Random Forest', y=1.01, fontsize=13)
_dbg_nc = sum(len(ax.collections) for ax in _shap_fig.axes)
print(f"  [DEBUG] Figura RF — ejes: {len(_shap_fig.axes)}, colecciones: {_dbg_nc} ({'TIENE DATOS' if _dbg_nc > 0 else 'SIN PUNTOS VISIBLES'})")
_fpath = os.path.abspath('shap_rf.png')
_shap_fig.savefig(_fpath, dpi=150, bbox_inches='tight')
print(f"  [DEBUG] Guardado: {_fpath} ({os.path.getsize(_fpath):,} bytes)")
plt.close('all')
try:
    os.startfile(_fpath)
except AttributeError:
    pass  # os.startfile solo existe en Windows

"""### FASE 15: ANÁLISIS DE VALIDACIÓN CRUZADA

### 4.6 Análisis de Validación Cruzada Temporal

Para evaluar la estabilidad y robustez de los modelos, se realizará una validación cruzada temporal utilizando `TimeSeriesSplit` con 5 divisiones (splits). Esto simula el progreso del tiempo, entrenando en datos pasados y validando en datos futuros.
"""

print("Configuring TimeSeriesSplit...")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Combine training and validation data for CV
X_train_val_combined = pd.concat([X_train, X_val], axis=0)
y_train_val_combined = pd.concat([y_train, y_val], axis=0)

# Ensure categorical features are correctly typed in the combined dataset for models that need it
cat_features_for_catboost = X_train_val_combined.select_dtypes(include='category').columns.tolist()
for col in cat_features_for_catboost:
    X_train_val_combined[col] = X_train_val_combined[col].astype('category')

print(f"Combined data shape for CV: {X_train_val_combined.shape}")

# Lists to store CV results
cv_results = {
    'CatBoost': {'RMSE': [], 'MAE': [], 'R2': [], 'MSE': []},
    'XGBoost': {'RMSE': [], 'MAE': [], 'R2': [], 'MSE': []},
    'LightGBM': {'RMSE': [], 'MAE': [], 'R2': [], 'MSE': []},
    'RandomForest': {'RMSE': [], 'MAE': [], 'R2': [], 'MSE': []}
}

"""#### Cross-Validation para CatBoost Regressor"""

print("Running TimeSeries CV for CatBoost...")
for fold, (train_index, test_index) in enumerate(tscv.split(X_train_val_combined)):
    print(f"  CatBoost - Fold {fold+1}/{n_splits}")
    X_cv_train, X_cv_test = X_train_val_combined.iloc[train_index], X_train_val_combined.iloc[test_index]
    y_cv_train, y_cv_test = y_train_val_combined.iloc[train_index], y_train_val_combined.iloc[test_index]

    cat_model_cv = CatBoostRegressor(
        iterations=study_catboost.best_params['iterations'],
        learning_rate=study_catboost.best_params['learning_rate'],
        depth=study_catboost.best_params['depth'],
        l2_leaf_reg=study_catboost.best_params['l2_leaf_reg'],
        bagging_temperature=study_catboost.best_params['bagging_temperature'],
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        cat_features=cat_features_for_catboost
    )
    cat_model_cv.fit(X_cv_train, y_cv_train, eval_set=(X_cv_test, y_cv_test), verbose=0)

    y_pred_cv = np.maximum(0, cat_model_cv.predict(X_cv_test))

    cv_results['CatBoost']['RMSE'].append(np.sqrt(mean_squared_error(y_cv_test, y_pred_cv)))
    cv_results['CatBoost']['MAE'].append(mean_absolute_error(y_cv_test, y_pred_cv))
    cv_results['CatBoost']['R2'].append(r2_score(y_cv_test, y_pred_cv))
    cv_results['CatBoost']['MSE'].append(mean_squared_error(y_cv_test, y_pred_cv))
print("CatBoost CV complete.")

"""#### Cross-Validation para XGBoost Regressor"""

print("Running TimeSeries CV for XGBoost...")
categorical_cols_ohe_xgb_cv = X_train_val_combined.select_dtypes(include='category').columns.tolist()

for fold, (train_index, test_index) in enumerate(tscv.split(X_train_val_combined)):
    print(f"  XGBoost - Fold {fold+1}/{n_splits}")
    X_cv_train, X_cv_test = X_train_val_combined.iloc[train_index], X_train_val_combined.iloc[test_index]
    y_cv_train, y_cv_test = y_train_val_combined.iloc[train_index], y_train_val_combined.iloc[test_index]

    # One-hot encode for XGBoost
    X_cv_train_ohe = pd.get_dummies(X_cv_train, columns=categorical_cols_ohe_xgb_cv, drop_first=True)
    X_cv_test_ohe = pd.get_dummies(X_cv_test, columns=categorical_cols_ohe_xgb_cv, drop_first=True)

    # Align columns: añadir columnas faltantes en test (valor 0) y reordenar igual que train
    for c in set(X_cv_train_ohe.columns) - set(X_cv_test_ohe.columns):
        X_cv_test_ohe[c] = 0
    X_cv_test_ohe = X_cv_test_ohe[X_cv_train_ohe.columns]

    # Convert to DMatrix
    dtrain_cv = xgb.DMatrix(X_cv_train_ohe, label=y_cv_train)
    dtest_cv = xgb.DMatrix(X_cv_test_ohe, label=y_cv_test)

    best_xgb_params_cv = study_xgboost.best_params.copy()
    best_xgb_params_cv['objective'] = 'reg:squarederror'
    best_xgb_params_cv['eval_metric'] = 'rmse'
    best_xgb_params_cv['seed'] = 42
    best_xgb_params_cv['nthread'] = -1
    num_boost_round_cv = best_xgb_params_cv.pop('n_estimators')

    xgb_model_cv = xgb.train(
        best_xgb_params_cv,
        dtrain_cv,
        num_boost_round=num_boost_round_cv,
    )

    y_pred_cv = xgb_model_cv.predict(dtest_cv)
    y_pred_cv = np.maximum(0, y_pred_cv) # Ensure non-negative predictions

    cv_results['XGBoost']['RMSE'].append(np.sqrt(mean_squared_error(y_cv_test, y_pred_cv)))
    cv_results['XGBoost']['MAE'].append(mean_absolute_error(y_cv_test, y_pred_cv))
    cv_results['XGBoost']['R2'].append(r2_score(y_cv_test, y_pred_cv))
    cv_results['XGBoost']['MSE'].append(mean_squared_error(y_cv_test, y_pred_cv))
print("XGBoost CV complete.")

"""#### Cross-Validation para LightGBM Regressor"""

print("Running TimeSeries CV for LightGBM...")
cat_features_for_lgbm_cv = X_train_val_combined.select_dtypes(include='category').columns.tolist()

for fold, (train_index, test_index) in enumerate(tscv.split(X_train_val_combined)):
    print(f"  LightGBM - Fold {fold+1}/{n_splits}")
    X_cv_train, X_cv_test = X_train_val_combined.iloc[train_index], X_train_val_combined.iloc[test_index]
    y_cv_train, y_cv_test = y_train_val_combined.iloc[train_index], y_train_val_combined.iloc[test_index]

    best_lgbm_params_cv = study_lgbm.best_params.copy()
    best_lgbm_params_cv['objective'] = 'regression_l1'
    best_lgbm_params_cv['metric'] = 'rmse'
    best_lgbm_params_cv['random_state'] = 42
    best_lgbm_params_cv['n_jobs'] = -1

    lgbm_model_cv = lgb.LGBMRegressor(**best_lgbm_params_cv)
    lgbm_model_cv.fit(X_cv_train, y_cv_train,
                    eval_set=[(X_cv_test, y_cv_test)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                    categorical_feature=cat_features_for_lgbm_cv)

    y_pred_cv = lgbm_model_cv.predict(X_cv_test)
    y_pred_cv = np.maximum(0, y_pred_cv) # Ensure non-negative predictions

    cv_results['LightGBM']['RMSE'].append(np.sqrt(mean_squared_error(y_cv_test, y_pred_cv)))
    cv_results['LightGBM']['MAE'].append(mean_absolute_error(y_cv_test, y_pred_cv))
    cv_results['LightGBM']['R2'].append(r2_score(y_cv_test, y_pred_cv))
    cv_results['LightGBM']['MSE'].append(mean_squared_error(y_cv_test, y_pred_cv))
print("LightGBM CV complete.")

"""#### Cross-Validation para Random Forest Regressor"""

print("Running TimeSeries CV for Random Forest...")
categorical_cols_ohe_rf_cv = X_train_val_combined.select_dtypes(include='category').columns.tolist()

for fold, (train_index, test_index) in enumerate(tscv.split(X_train_val_combined)):
    print(f"  Random Forest - Fold {fold+1}/{n_splits}")
    X_cv_train, X_cv_test = X_train_val_combined.iloc[train_index], X_train_val_combined.iloc[test_index]
    y_cv_train, y_cv_test = y_train_val_combined.iloc[train_index], y_train_val_combined.iloc[test_index]

    # One-hot encode for Random Forest
    X_cv_train_ohe = pd.get_dummies(X_cv_train, columns=categorical_cols_ohe_rf_cv, drop_first=True)
    X_cv_test_ohe = pd.get_dummies(X_cv_test, columns=categorical_cols_ohe_rf_cv, drop_first=True)

    # Align columns: añadir columnas faltantes en test (valor 0) y reordenar igual que train
    for c in set(X_cv_train_ohe.columns) - set(X_cv_test_ohe.columns):
        X_cv_test_ohe[c] = 0
    X_cv_test_ohe = X_cv_test_ohe[X_cv_train_ohe.columns]

    best_rf_params_cv = study_rf.best_params.copy()
    best_rf_params_cv['random_state'] = 42
    best_rf_params_cv['n_jobs'] = -1

    rf_model_cv = RandomForestRegressor(**best_rf_params_cv)
    rf_model_cv.fit(X_cv_train_ohe, y_cv_train)

    y_pred_cv = rf_model_cv.predict(X_cv_test_ohe)
    y_pred_cv = np.maximum(0, y_pred_cv) # Ensure non-negative predictions

    cv_results['RandomForest']['RMSE'].append(np.sqrt(mean_squared_error(y_cv_test, y_pred_cv)))
    cv_results['RandomForest']['MAE'].append(mean_absolute_error(y_cv_test, y_pred_cv))
    cv_results['RandomForest']['R2'].append(r2_score(y_cv_test, y_pred_cv))
    cv_results['RandomForest']['MSE'].append(mean_squared_error(y_cv_test, y_pred_cv))
print("Random Forest CV complete.")

"""#### Visualización de los Resultados de la Validación Cruzada"""

print("Generating CV results plots...")

# Prepare data for plotting
plot_data = []
for model_name, metrics in cv_results.items():
    for metric_name, values in metrics.items():
        for value in values:
            plot_data.append({'Model': model_name, 'Metric': metric_name, 'Value': value})

df_plot_cv = pd.DataFrame(plot_data)

fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=False)
fig.suptitle('Resultados de la Validación Cruzada Temporal (5 Folds)', fontsize=16, weight='bold', y=1.02)

metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MSE']
titles = {
    'RMSE': 'RMSE (Menor es Mejor)',
    'MAE': 'MAE (Menor es Mejor)',
    'R2': 'R² (Mayor es Mejor)',
    'MSE': 'MSE (Menor es Mejor)'
}
palettes = {
    'RMSE': 'Oranges_r',
    'MAE': 'Greens_r',
    'R2': 'Blues_r',
    'MSE': 'Reds_r'
}

for i, metric in enumerate(metrics_to_plot):
    subset_data = df_plot_cv[df_plot_cv['Metric'] == metric]
    sns.boxplot(ax=axes[i], x='Model', y='Value', data=subset_data, palette=palettes[metric])
    axes[i].set_title(titles[metric])
    axes[i].set_xlabel('')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("CV analysis complete.")


"""### 4.7 Evaluación de Resiliencia Operativa Predicha — Modelo Ganador (LightGBM)"""

print("\n" + "="*60)
print("  4.7 EVALUACIÓN DE RESILIENCIA OPERATIVA PREDICHA")
print("  Modelo ganador: LightGBM (mejor MAE, RMSE, R² y MSE)")
print("="*60)

# RO real del conjunto de prueba
y_test_ro = 1 / (1 + y_test.values)

# RO predicha por LightGBM (clipping para evitar negativos)
y_pred_lgbm_ro = 1 / (1 + np.maximum(0, y_pred_lgbm))

# --- Métricas MAE y RMSE sobre escala RO (0–1) ---
mae_ro_lgbm  = mean_absolute_error(y_test_ro, y_pred_lgbm_ro)
rmse_ro_lgbm = np.sqrt(mean_squared_error(y_test_ro, y_pred_lgbm_ro))

print("\n--- Métricas de error sobre Resiliencia Operativa (escala 0–1) ---")
print(f"  MAE  (RO): {mae_ro_lgbm:.6f}")
print(f"  RMSE (RO): {rmse_ro_lgbm:.6f}")

# --- Scatter plot: RO real vs RO predicha ---
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test_ro, y_pred_lgbm_ro, alpha=0.35, color='steelblue',
           edgecolors='none', s=18, label='Observaciones test')
lims = [min(y_test_ro.min(), y_pred_lgbm_ro.min()),
        max(y_test_ro.max(), y_pred_lgbm_ro.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Predicción perfecta')
ax.set_xlabel('RO Real', fontsize=11)
ax.set_ylabel('RO Predicha — LightGBM', fontsize=11)
ax.set_title('Resiliencia Operativa: Real vs Predicha\nLightGBM (modelo ganador)',
             fontsize=13, weight='bold')
ax.annotate(f'MAE  = {mae_ro_lgbm:.5f}\nRMSE = {rmse_ro_lgbm:.5f}',
            xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10, color='darkred',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))
ax.legend()
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# --- Categorización y Accuracy ---
# Umbrales calculados sobre el conjunto de entrenamiento (sin data leakage)
y_train_ro = 1 / (1 + y_train.values)
p33_train = np.percentile(y_train_ro, 33)
p66_train = np.percentile(y_train_ro, 66)

print(f"\n  Umbrales de categorización (train set):")
print(f"    p33 = {p33_train:.4f}  →  RO ≤ p33: Baja")
print(f"    p66 = {p66_train:.4f}  →  RO ≤ p66: Media  |  RO > p66: Alta")

def categorize_ro(ro_array, p33, p66):
    return np.where(ro_array <= p33, 'Baja',
           np.where(ro_array <= p66, 'Media', 'Alta'))

y_test_cat  = categorize_ro(y_test_ro,       p33_train, p66_train)
y_lgbm_cat  = categorize_ro(y_pred_lgbm_ro,  p33_train, p66_train)

acc_lgbm = accuracy_score(y_test_cat, y_lgbm_cat)
print(f"\n--- Accuracy de Categoría (Baja / Media / Alta) ---")
print(f"  LightGBM: {acc_lgbm:.4f} ({acc_lgbm*100:.2f}%)")

# --- Matriz de confusión ---
labels = ['Alta', 'Baja', 'Media']
fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle('Matriz de Confusión — LightGBM\nCategoría de Resiliencia Operativa',
             fontsize=13, weight='bold')
cm = confusion_matrix(y_test_cat, y_lgbm_cat, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Accuracy: {acc_lgbm*100:.2f}%')
plt.tight_layout()
plt.show()


"""### PRONÓSTICO 3 MESES — LightGBM (Modelo Ganador)"""

print("\n" + "="*60)
print("  PRONÓSTICO 3 MESES — LightGBM (modelo ganador)")
print("  Forecasting recursivo: cada mes predicho alimenta el siguiente")
print("="*60)

# Reconstruir fecha desde Year+Month
df_hist_fc = df_sorted_fe.copy()
df_hist_fc['_fecha'] = pd.to_datetime(
    df_hist_fc['Year'].astype(str) + '-' + df_hist_fc['Month'].astype(str) + '-01'
)
last_date = df_hist_fc['_fecha'].max()

# Empresa con más registros en el dataset
empresa_fc = df_hist_fc['Empresa_operadora_grouped'].value_counts().index[0]
print(f"  Último mes del dataset: {last_date.strftime('%Y-%m')}")
print(f"  Empresa seleccionada para el pronóstico: {empresa_fc}")

feature_cols = X_train.columns.tolist()

# RO mensual promedio real del test set — solo empresa seleccionada
df_test_ro_plot = df_sorted_fe.iloc[val_split_point:].copy()
df_test_ro_plot = df_test_ro_plot[
    df_test_ro_plot['Empresa_operadora_grouped'] == empresa_fc
].copy()
df_test_ro_plot['RO'] = 1 / (1 + df_test_ro_plot['Num_Reclamos_por_averia_Aggregated'])
df_test_ro_plot['_fecha'] = pd.to_datetime(
    df_test_ro_plot['Year'].astype(str) + '-' + df_test_ro_plot['Month'].astype(str) + '-01'
)
ro_mensual_real = df_test_ro_plot.groupby('_fecha')['RO'].mean().reset_index()

# Forecasting recursivo: T+1 → T+2 → T+3 — solo grupos de la empresa seleccionada
forecast_records = []

for i in range(1, 4):
    future_date     = last_date + pd.DateOffset(months=i)
    future_year     = future_date.year
    future_month    = future_date.month
    future_quarter  = (future_month - 1) // 3 + 1
    future_semester = (future_month - 1) // 6 + 1

    grupos = df_hist_fc[
        df_hist_fc['Empresa_operadora_grouped'] == empresa_fc
    ][grouping_keys].drop_duplicates()
    rows = []

    for _, grp in grupos.iterrows():
        mask = np.ones(len(df_hist_fc), dtype=bool)
        for k in grouping_keys:
            mask &= (df_hist_fc[k] == grp[k]).values
        vals = df_hist_fc[mask].sort_values('_fecha')['Num_Reclamos_por_averia_Aggregated'].values

        def get_lag(n):
            return float(vals[-n]) if len(vals) >= n else -1.0

        def get_rolling_mean(w):
            v = vals[-w:] if len(vals) >= w else vals
            return float(v.mean()) if len(v) > 0 else -1.0

        def get_rolling_std(w):
            v = vals[-w:] if len(vals) >= w else vals
            return float(v.std()) if len(v) > 1 else -1.0

        row = {k: grp[k] for k in grouping_keys}
        row.update({
            'Year': future_year, 'Month': future_month,
            'Quarter': future_quarter, 'Semester': future_semester,
            'lag_1':  get_lag(1),  'lag_3':  get_lag(3),
            'lag_6':  get_lag(6),  'lag_12': get_lag(12),
            'rolling_mean_3':  get_rolling_mean(3),
            'rolling_mean_6':  get_rolling_mean(6),
            'rolling_mean_12': get_rolling_mean(12),
            'rolling_std_3':   get_rolling_std(3),
            'rolling_std_6':   get_rolling_std(6),
            'rolling_std_12':  get_rolling_std(12),
        })
        rows.append(row)

    df_future = pd.DataFrame(rows)

    # Restaurar dtypes categóricos con las mismas categorías que vio LightGBM en entrenamiento
    for col in categorical_cols_native:
        if col in df_future.columns and col in X_train.columns:
            df_future[col] = pd.Categorical(df_future[col],
                                            categories=X_train[col].cat.categories)

    y_future_pred = np.maximum(0, lgbm_model.predict(df_future[feature_cols]))
    ro_pred   = 1 / (1 + y_future_pred)
    ro_mean   = ro_pred.mean()
    categoria = categorize_ro(np.array([ro_mean]), p33_train, p66_train)[0]
    print(f"  {future_date.strftime('%Y-%m')}: RO promedio = {ro_mean:.4f}  → {categoria}")

    forecast_records.append({
        '_fecha': future_date.replace(day=1),
        'RO': ro_mean,
        'categoria': categoria
    })

    # Agregar predicciones al historial para el siguiente mes (paso recursivo)
    df_future['Num_Reclamos_por_averia_Aggregated'] = y_future_pred
    df_future['_fecha'] = future_date.replace(day=1)
    cols_comunes = [c for c in df_hist_fc.columns if c in df_future.columns]
    df_hist_fc = pd.concat([df_hist_fc, df_future[cols_comunes]], ignore_index=True)

df_forecast = pd.DataFrame(forecast_records)

# --- Gráfico: RO real (test) + pronóstico 3 meses ---
plt.figure(figsize=(13, 5))
plt.plot(ro_mensual_real['_fecha'], ro_mensual_real['RO'],
         marker='o', color='steelblue', linewidth=2, label='RO Real (test set)')
plt.plot(df_forecast['_fecha'], df_forecast['RO'],
         marker='s', linestyle='--', color='tomato', linewidth=2, label='Pronóstico LightGBM')
plt.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5, label='Fin del dataset')

for _, r in df_forecast.iterrows():
    plt.annotate(f"{r['RO']:.3f}\n({r['categoria']})",
                 (r['_fecha'], r['RO']),
                 textcoords='offset points', xytext=(0, 14),
                 ha='center', fontsize=9, color='tomato', weight='bold')

plt.title(f'Resiliencia Operativa — Real vs Pronóstico 3 Meses\n{empresa_fc} · LightGBM (modelo ganador)',
          fontsize=13, weight='bold')
plt.xlabel('Mes')
plt.ylabel('RO Promedio (escala 0–1)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()