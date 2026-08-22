import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# ==============================================================================
# 1. TRANSFORMADORES PERSONALIZADOS (La materia prima)
# ==============================================================================

class SafeLog1pTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica np.log1p dinámicamente solo a variables numéricas, no binarias y positivas.
    """
    def __init__(self):
        self.cols_to_transform_ = []

    def fit(self, X, y=None):
        self.cols_to_transform_ = [] # Reinicia la memoria en cada fit
        numericas = X.select_dtypes(include=[np.number]).columns
        for col in numericas:
            valor_minimo = X[col].min()
            es_binaria = X[col].isin([0, 1, np.nan]).all()
            if valor_minimo >= 0 and not es_binaria:
                self.cols_to_transform_.append(col)
        return self
    
    def transform(self, X):
        # Aplicar la transformación np.log1p sobre copias para mantener la inmutabilidad
        X_out = X.copy()
        for col in self.cols_to_transform_:
            if col in X_out.columns:
                X_out[col] = np.log1p(X_out[col])
        return X_out
        
    def get_feature_names_out(self, input_features=None):
        return input_features

class ExplicitCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Inicializa el transformador con las reglas de negocio descubiertas en el EDA.
    :param categories_to_keep: Diccionario con formato {'NombreColumna': ['Cat1', 'Cat2']}
    """
    def __init__(self, categories_to_keep):
        self.categories_to_keep = categories_to_keep

    def fit(self, X, y=None):
        # Como nuestras reglas son explícitas y derivadas de un EDA manual riguroso,
        # no necesitamos calcular frecuencias aquí. Solo retornamos el objeto.
        return self

    def transform(self, X):
        # Creamos una copia para no alterar el DataFrame original en memoria
        X_transformed = X.copy()
        
        # Iteramos sobre las reglas que le pasamos al instanciar la clase
        for col, kept_cats in self.categories_to_keep.items():
            if col in X_transformed.columns:
                # Aplicamos la lógica: Si el valor está en la lista de aprobados, lo dejamos.
                # Si no, lo mandamos a 'Other' o al nombre de agrupación específico.
                
                # Manejo especial para HouseStyle que acordamos llamar 'Rare_Styles'
                other_label = 'Rare_Styles' if col == 'HouseStyle' else 'Other'
                
                X_transformed[col] = X_transformed[col].apply(
                    lambda val: val if val in kept_cats else other_label
                )
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class TargetKMeansClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.mapping_ = {}
        self.default_cluster_ = 1  # Por defecto (Promedio) para barrios desconocidos
        
    def fit(self, X, y):
        # 1. Unimos el barrio y el precio en un DataFrame temporal
        # Ojo: X aquí es un DataFrame de una sola columna ('Neighborhood')
        col_name = X.columns[0]
        df_temp = pd.DataFrame({col_name: X[col_name], 'target': y})
        
        # 2. Calculamos el precio medio de cada barrio (solo con datos de Train)
        means = df_temp.groupby(col_name)['target'].mean().reset_index()
        
        # 3. Entrenamos K-Means sobre esos precios medios
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        means['raw_cluster'] = kmeans.fit_predict(means[['target']])
        
        # 4. ORDENAMIENTO CRÍTICO: K-Means asigna números al azar (el cluster 0 podría ser el más caro).
        # Debemos ordenar los clusters por su precio promedio para que 0 < 1 < 2
        cluster_centers = means.groupby('raw_cluster')['target'].mean().sort_values()
        
        # Creamos un diccionario para renombrar los clusters al orden correcto
        order_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_centers.index)}
        means['ordered_cluster'] = means['raw_cluster'].map(order_mapping)
        
        # 5. Guardamos el diccionario final {NombreBarrio: ClusterOrdinal}
        self.mapping_ = dict(zip(means[col_name], means['ordered_cluster']))
        
        # Guardamos el cluster promedio (1) para imputar barrios que aparezcan en Test pero no en Train
        self.default_cluster_ = 1 
        return self

    def transform(self, X):
        X_transformed = X.copy()
        col_name = X_transformed.columns[0]
        
        # Mapeamos los barrios a sus clusters. Si no existe, va al cluster por defecto.
        X_transformed[col_name] = X_transformed[col_name].map(self.mapping_).fillna(self.default_cluster_)
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        return input_features

class UniversalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_functions):
        self.feature_functions = feature_functions
        
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for feature_name, func in self.feature_functions.items():
            X_out[f'feature_engineered__{feature_name}'] = func(X_out)
        return X_out

    def get_feature_names_out(self, input_features=None):
        nuevas_cols = [f'feature_engineered__{name}' for name in self.feature_functions.keys()]
        if input_features is not None:
            return list(input_features) + nuevas_cols
        elif hasattr(self, "feature_names_in_"):
            return list(self.feature_names_in_) + nuevas_cols
        else:
            return nuevas_cols

from xgboost import XGBRegressor

class FScoreFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Entrena un XGBoost interno para evaluar la importancia de las variables (F-Score).
    Memoriza qué columnas superan el umbral y poda el resto en el método transform().
    """
    def __init__(self, threshold=2, random_state=42):
        self.threshold = threshold
        self.random_state = random_state
        self.features_to_keep_ = []

    def fit(self, X, y):
        # 1. Entrenamos un modelo evaluador silencioso
        evaluator = XGBRegressor(random_state=self.random_state, n_jobs=-1)
        evaluator.fit(X, y)

        # 2. Extraemos los F-Scores
        booster = evaluator.get_booster()
        f_scores = booster.get_score(importance_type='weight')

        # 3. Guardamos en memoria solo las columnas que superan el umbral
        self.features_to_keep_ = [
            col for col in X.columns 
            if int(f_scores.get(col, 0)) > self.threshold
        ]
        
        #print(f"✂️ Feature Selection: De {X.shape[1]} variables, se conservan {len(self.features_to_keep_)}.")
        return self

    def transform(self, X):
        # Filtramos el DataFrame para devolver solo las columnas ganadoras
        return X[self.features_to_keep_].copy()

    def get_feature_names_out(self, input_features=None):
        return self.features_to_keep_

# ==============================================================================
# 2. FUNCIONES DE INGENIERÍA Y CONSTANTES DE NEGOCIO
# ==============================================================================

def calc_propiedad_riesgo(df):
    """
    Genera un flag de riesgo inmobiliario basado en el cruce de variables
    post-preprocesamiento, mitigando sesgos aditivos del modelo.
    """
    # 1. Definición explícita de nombres (obtenidos de preprocesador.get_feature_names_out())
    # Esto garantiza la trazabilidad absoluta del pipeline
    col_qual = 'remainder__overall_qual'
    col_neigh = 'neighborhood_kmeans__neighborhood' 
    col_paved = 'remainder__paved_drive'
    
    if not all(c in df.columns for c in [col_qual, col_neigh, col_paved]):
        return np.zeros(len(df)) # Fallback de seguridad en producción
        
    # 3. Aplicación de reglas de negocio con tipos de datos nativos
    es_baja_calidad = df[col_qual].astype(float) < 5
    es_barrio_economico = df[col_neigh] == 0  # 0 mapea a 'pobre' según el target encoding
    es_sin_pavimento = df[col_paved] == 0     # Evaluación estrictamente numérica post-encoding 
    
    # Conjunción de condiciones
    condicion_riesgo = (es_baja_calidad & es_barrio_economico) | es_sin_pavimento
    
    # Retorna un vector binario optimizado para XGBoost
    return np.where(condicion_riesgo, 1, 0)

REGLAS_AGRUPACION = {
    'sale_condition': ['Normal', 'Abnorml', 'Partial'],
    'sale_type': ['WD', 'New'],
    'exterior1st': ['MetalSd', 'HdBoard', 'Plywood'],
    'house_style': ['1Story', '2Story', '1.5Fin', 'SLvl']
}

MIS_NUEVAS_REGLAS = {
    # 'es_propiedad_riesgo': calc_propiedad_riesgo
}

# ==============================================================================
# 3. LA FÁBRICA DEL PIPELINE (El orquestador de variables)
# ==============================================================================

def build_preprocessor() -> Pipeline:
    """
    Construye y retorna el pipeline maestro inmutable.
    Al llamarlo desde train.py, generará un objeto limpio listo para hacer .fit()
    """
    # 1. Definición de grupos de columnas
    
    # Variables que van directo a OHE (baja cardinalidad)
    direct_ohe_cols = ['land_contour', 'lot_config', 'bldg_type', 'foundation', 'garage_type', 'mas_vnr_type', 'roof_style', 'fence']
    
    # Variables que requieren limpieza previa (alta cardinalidad)
    complex_nominal_cols = ['sale_condition', 'sale_type', 'exterior1st', 'house_style']
    
    # Lista exclusiva para el Target Encoding y K-Means
    neighborhood_col = ['neighborhood']
    
    # 2. Sub-pipelines
    complex_nominal_pipeline = Pipeline(steps=[
        ('grouper', ExplicitCategoryGrouper(categories_to_keep=REGLAS_AGRUPACION)),
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # 3. Enrutador Base
    preprocesador_base = ColumnTransformer(
        transformers=[
            ('simple_nominal', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), direct_ohe_cols),
            ('complex_nominal', complex_nominal_pipeline, complex_nominal_cols),
            ('neighborhood_kmeans', TargetKMeansClusterer(n_clusters=3), neighborhood_col)
        ],
        remainder='passthrough' 
    )
    
    #preprocesador_base.set_output(transform="pandas")
    
    # 2. La secuencia estricta: Primero transformar, LUEGO podar
    pipeline_maestro = Pipeline(steps=[
        ('logaritmo_seguro', SafeLog1pTransformer()),
        ('transformacion_base', preprocesador_base),
        #('seleccion_fscore', FScoreFeatureSelector(threshold=2)) # Actúa sobre el 100% de la matriz ya transformada
    ])
    
    pipeline_maestro.set_output(transform="pandas")
    
    
    return pipeline_maestro