# 🏠 Ames Real Estate Pricing Engine | End-to-End ML Architecture

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-%23178C3A.svg?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-FF694B?logo=dbt&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-blue?logo=optuna&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-669DF6?logo=google-bigquery&logoColor=white)

## 📌 Visión General del Negocio

Este proyecto implementa un motor de predicción de precios inmobiliarios altamente optimizado. Superando el enfoque tradicional de cuadernos de análisis estáticos, el sistema está diseñado como una arquitectura de datos completa (**End-to-End**), abarcando desde la ingesta y transformación en Data Warehouse hasta el despliegue mediante una **API RESTful**.

> **🎯 Objetivo Central:** Proporcionar valoraciones precisas de mercado minimizando la varianza del error y controlando la asimetría extrema típica de los datos financieros.

### 📊 Métricas de Impacto

| Métrica | Resultado |
| :--- | :--- |
| **Kaggle Benchmark (RMSLE)** | `[0.12620]` -> Top 24% Overall with only 1 optimize model|
| **Margen de Error Promedio** | `[12%]` <br> *(~ `$20,263` sobre una propiedad promedio)* |

---

## 🏗️ Arquitectura del Sistema Full-Stack

El proyecto adopta un enfoque modular que separa la ingeniería de datos del modelado predictivo, previniendo activamente el *Data Leakage*.

*   **🗄️ Capa de Datos (ELT):** [Repositorio de dbt aquí](#) *(<- Agrega el link a tu repo de dbt)*
    *   Procesamiento ejecutado en **Google BigQuery**.
    *   Limpieza, estandarización e imputación base gestionada con **dbt**.
*   **🧠 Capa de Machine Learning:**
    *   *Feature Engineering* dinámico orientado a negocio (ej. Ratios de rentabilidad, segmentación de amenidades).
    *   Algoritmo core: **XGBoost Regressor** con ajuste bayesiano (**Optuna**).
*   **🌐 Capa de Servicio:**
    *   Inferencia en tiempo real expuesta mediante **FastAPI**.
    *   Validación estricta de payloads con **Pydantic**.

---

## ⚙️ Hitos Técnicos y Rigor Analítico

Para asegurar la robustez del modelo en producción, se implementaron las siguientes estrategias avanzadas:

- **📈 Mitigación de Asimetría (*Skewness*):** Aplicación estructurada de transformaciones `np.log1p` sobre el vector objetivo y predictores de cola larga, recuperando resolución matemática en el algoritmo de agrupamiento de XGBoost (`tree_method='hist'`).
- **🗺️ Clustering Espacial:** Uso de `TargetKMeansClusterer` para reducir la alta cardinalidad de los vecindarios a 3 macro-zonas de valor económico, sin exponer promedios directos al modelo predictivo.
- **✂️ Selección de Características Determinista:** Poda automatizada de variables ruidosas mediante la extracción de *F-scores* (Frecuencia de división) del *Booster*, eliminando dimensiones con impacto nulo ($F \le 2$).
- **🎯 Optimización Bayesiana Segura:** Búsqueda de hiperparámetros con **Optuna** + *Early Stopping* sobre una matriz estrictamente podada, evaluada mediante un bucle de Validación Cruzada *Out-of-Fold* (OOF).

<!-- 💡 CONSEJO: Descomenta este bloque e inserta las rutas de tus imágenes (SHAP values, Optuna History) para darle un impacto visual tremendo al portfolio -->
<!-- 
<p align="center">
  <img src="ruta/a/tu/grafico_shap.png" width="45%" alt="Análisis de SHAP Values">
  <img src="ruta/a/tu/optuna_history.png" width="45%" alt="Historia de Optimización con Optuna">
</p>
-->

---

## 📂 Estructura del Proyecto

Adhiriendo a los estándares de *Cookiecutter Data Science*:

```text
ames_pricing_system/
├── api/                     # Capa de servicio RESTful
│   ├── main.py              # Endpoints de FastAPI
│   └── schemas.py           # Validaciones de entrada de datos (Pydantic)
│
├── notebooks/               # Experimentación y EDA
│   ├── 01_eda_and_drift_audit.ipynb 
│   └── 02_model_development.ipynb    
│
├── src/                     # Código fuente de producción (Pipeline OOP)
│   ├── features.py          # Clases personalizadas (UniversalFeatureEngineer)
│   ├── data_ingestion.py    # Conectores y splitters (Muro de Hierro de Validación)
│   └── modeling.py          # Pipeline maestro y evaluadores
│
├── models/                  # Artefactos serializados (.joblib)
├── requirements.txt         # Dependencias del entorno
└── README.md                # Documentación del sistema
```

# 📈 Estrategia de Escalamiento a Big Data (Pragmatismo Industrial)

La arquitectura actual de este repositorio prioriza la integridad estadística estricta (Cero Data Leakage) ejecutando todas las transformaciones espacial-temporales y la poda por $F\text{-score}$ dentro de cada iteración del K-Fold. Si bien este enfoque es ideal estadísticamente para datasets pequeños (~1,400 registros), aplicar esta recursividad a un entorno transaccional real de más de 5 millones de registros generaría un overhead computacional insostenible.

Para desplegar este motor de precios en un entorno masivo corporativo, la arquitectura debe pivotar hacia el pragmatismo computacional mediante tres pilares de optimización:

## 1. Shift-Left Analytics
El motor de Python (Scikit-Learn) no debe utilizar su memoria RAM para cálculos deterministas. En un entorno de Big Data, todas las transformaciones estáticas se delegarian hacia atrás en la arquitectura, directamente al Data Warehouse.
- Impacto: Python solo ingiere la matriz pre-procesada y reserva su capacidad de cómputo exclusivamente para operaciones dinámicas dependientes del Target que SQL no puede manejar eficientemente (Target Encoding cruzado, Clustering K-Means de coordenadas y entrenamiento del XGBoost).

## Poda Asíncrona (Proxy Sampling)

Recalcular el algoritmo FScoreFeatureSelector dentro del pipeline de optimización bayesiana en millones de registros multiplica el costo de hardware por cada prueba.

- Se extrae una muestra estratificada representativa (ej. 150,000 registros). Se entrena un modelo XGBoost aislado sobre esta muestra para extraer los $F\text{-scores}$, identificando las variables ruidosas ($F \le 2$). Esta lista negra de columnas se congela.
- Impacto: El pipeline de producción principal ya no calcula los scores dinámicamente; simplemente aplica un filtro estático de exclusión basado en la lista congelada, liberando a Optuna para buscar hiperparámetros en el espacio limpio de forma ultrarrápid

## Transición de K-Fold a Hold-Out Validation
La validacion cruzada es una herramienta diseñada para mitigar la alta varianza estadísitca inherente en datasets.

- Implementación: Apoyándonos en la Ley de los Grandes Números, al superar el umbral del medio millón de registros, la varianza estadística de las particiones se estabiliza. Reemplazamos el costoso 5-Fold CV por una única partición estructurada Hold-out (ej. 80% Train, 10% Validation para Early Stopping, 10% Test).
- Impacto: Reducción inmediata del 80% en los tiempos de cómputo y costos de Cloud Computing en la fase de ajuste de Optuna, manteniendo una métrica de generalización (RMSLE) altamente confiable.

# 🚀 Cómo ejecutar el proyecto
1. Entorno Virtual y Dependencias
Bash


### 1. Requisitos Previos
Asegúrate de tener Python instalado en tu sistema.

### 2. Instalar Poetry (si no lo tienes)
Si aún no tienes Poetry instalado en tu computadora, ejecuta el siguiente comando:

**En Windows (PowerShell):**
```bash
(Invoke-WebRequest -Uri https://python-poetry.org -UseBasicParsing).Content | py -
```

**En Linux / macOS:**
```bash
curl -sSL https://python-poetry.org | python3 -
```

### 3. Crear e Instalar el Entorno Virtual
Dirígete a la carpeta raíz del proyecto (donde está el archivo `pyproject.toml`) e instala todas las dependencias del proyecto:

```bash
poetry install
```
*Nota: Este comando creará automáticamente el entorno virtual e instalará librerías como pandas, scikit-learn, etc.*

### 4. Activar el Entorno Virtual
Para entrar al entorno virtual y ejecutar tus scripts de Python, usa:

```bash
poetry shell
```

O si prefieres ejecutar un script directamente sin activar la terminal completa:
```bash
poetry run python src/tu_script.py
```

## 5. Entrenamiento del Modelo
Para replicar el preprocesamiento, ejecutar la poda por F-score, afinar hiperparámetros y generar el artefacto .joblib en la carpeta models/:

```bash
python src/train.py
```
### 6. Levantar la API de Predicción
```Bash
cd api
uvicorn main:app --reload
```

💡 Nota: La documentación interactiva de la API (Swagger UI) estará disponible automáticamente en http://localhost:8000/docs. Allí podrás probar predicciones en tiempo real.

Diseñado y desarrollado para demostrar rigor en analítica predictiva y arquitectura de datos.
