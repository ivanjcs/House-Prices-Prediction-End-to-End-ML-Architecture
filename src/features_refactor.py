import os
import json
import warnings
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# Importaciones locales supuestas
# from data_ingestion import get_bq_client, get_train_data

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. TRANSFORMADORES PERSONALIZADOS (Limpios y Corregidos)
# ==============================================================================

class SafeLog1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_transform_ = []

    def fit(self, X, y=None):
        self.cols_to_transform_ = [] 
        numericas = X.select_dtypes(include=[np.number]).columns
        for col in numericas:
            valor_minimo = X[col].min()
            es_binaria = X[col].isin([0, 1, np.nan]).all()
            if valor_minimo >= 0 and not es_binaria:
                self.cols_to_transform_.append(col)
        return self
    
    def transform(self, X):
        X_out = X.copy()
        for col in self.cols_to_transform_:
            if col in X_out.columns:
                X_out[col] = np.log1p(X_out[col])
        return X_out
        
    def get_feature_names_out(self, input_features=None):
        return input_features


class ExplicitCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, categories_to_keep):
        self.categories_to_keep = categories_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, kept_cats in self.categories_to_keep.items():
            if col in X_transformed.columns:
                # CORRECCIÓN BUG SILENCIOSO: 'house_style' en minúsculas
                other_label = 'Rare_Styles' if col == 'house_style' else 'Other'
                
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
        self.default_cluster_ = 1  
        
    def fit(self, X, y):
        col_name = X.columns[0]
        # CORRECCIÓN DE SEGURIDAD: y.values para evitar desalineación de índices
        df_temp = pd.DataFrame({col_name: X[col_name], 'target': y.values})
        
        means = df_temp.groupby(col_name)['target'].mean().reset_index()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        means['raw_cluster'] = kmeans.fit_predict(means[['target']])
        
        cluster_centers = means.groupby('raw_cluster')['target'].mean().sort_values()
        order_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_centers.index)}
        means['ordered_cluster'] = means['raw_cluster'].map(order_mapping)
        
        self.mapping_ = dict(zip(means[col_name], means['ordered_cluster']))
        self.default_cluster_ = 1 
        return self

    def transform(self, X):
        X_transformed = X.copy()
        col_name = X_transformed.columns[0]
        X_transformed[col_name] = X_transformed[col_name].map(self.mapping_).fillna(self.default_cluster_)
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        return input_features


class FScoreFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=2, random_state=42):
        self.threshold = threshold
        self.random_state = random_state
        self.features_to_keep_ = []

    def fit(self, X, y):
        evaluator = XGBRegressor(random_state=self.random_state, n_jobs=-1)
        evaluator.fit(X, y)

        booster = evaluator.get_booster()
        f_scores = booster.get_score(importance_type='weight')

        self.features_to_keep_ = [
            col for col in X.columns 
            if int(f_scores.get(col, 0)) > self.threshold
        ]
        return self

    def transform(self, X):
        # Filtro de seguridad por si una columna no existe al transformar
        columnas_seguras = [col for col in self.features_to_keep_ if col in X.columns]
        return X[columnas_seguras].copy()

    def get_feature_names_out(self, input_features=None):
        return self.features_to_keep_


# ==============================================================================
# 2. EL PIPELINE MAESTRO (Orquestador)
# ==============================================================================

REGLAS_AGRUPACION = {
    'sale_condition': ['Normal', 'Abnorml', 'Partial'],
    'sale_type': ['WD', 'New'],
    'exterior1st': ['MetalSd', 'HdBoard', 'Plywood'],
    'house_style': ['1Story', '2Story', '1.5Fin', 'SLvl']
}

def build_preprocessor() -> Pipeline:
    direct_ohe_cols = ['land_contour', 'lot_config', 'bldg_type', 'foundation', 'garage_type', 'mas_vnr_type', 'roof_style', 'fence']
    complex_nominal_cols = ['sale_condition', 'sale_type', 'exterior1st', 'house_style']
    neighborhood_col = ['neighborhood']
    
    complex_nominal_pipeline = Pipeline(steps=[
        ('grouper', ExplicitCategoryGrouper(categories_to_keep=REGLAS_AGRUPACION)),
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocesador_base = ColumnTransformer(
        transformers=[
            ('simple_nominal', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), direct_ohe_cols),
            ('complex_nominal', complex_nominal_pipeline, complex_nominal_cols),
            ('neighborhood_kmeans', TargetKMeansClusterer(n_clusters=3), neighborhood_col)
        ],
        remainder='passthrough' 
    )
    
    pipeline_maestro = Pipeline(steps=[
        ('logaritmo_seguro', SafeLog1pTransformer()),
        ('transformacion_base', preprocesador_base),
        ('seleccion_fscore', FScoreFeatureSelector(threshold=2))
    ])
    
    pipeline_maestro.set_output(transform="pandas")
    return pipeline_maestro