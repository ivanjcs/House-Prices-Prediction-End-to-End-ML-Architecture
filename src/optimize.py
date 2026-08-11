import os
import json
import warnings
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error

# Importaciones locales
from data_ingestion import get_bq_client, get_train_data
from features import build_preprocessor

warnings.filterwarnings('ignore')

def objective(trial, X, y):
    """
    Función objetivo para Optuna.
    Recibe X e y crudos (del train set). El preprocesamiento ocurre DENTRO del KFold.
    """
    param_grid = {
        # Arquitectura del árbol
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
        
        # Muestreo (Previene memorización)
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # Regularización Matemática (Freno de mano al sobreajuste)
        # alpha (L1) actúa como un machete: apaga variables inútiles llevándolas a cero.
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 50.0, log=True),
        # lambda (L2) actúa como una camisa de fuerza: evita que los pesos se hagan muy grandes.
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        
        # Poda Proactiva
        # gamma exige una reducción mínima de error para permitir que una hoja se divida.
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    
    # 2. EL BUCLE STRICTO (Cero Data Leakage)
    for train_idx, val_idx in kf.split(X):
        
        preprocesador = build_preprocessor()
        
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        # El pipeline aprende de X_tr y transforma X_tr
        X_tr_procesado = preprocesador.fit_transform(X_tr, y_tr)
        
        # El pipeline transforma X_va a ciegas (basado en lo aprendido en X_tr)
        X_va_procesado = preprocesador.transform(X_va)
        
        # 3. XGBoost con Early Stopping
        xgb_model = XGBRegressor(
            **param_grid,
            learning_rate=0.05,
            n_estimators=2000,
            early_stopping_rounds=50,
            eval_metric='rmse',
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(
            X_tr_procesado, y_tr,
            eval_set=[(X_va_procesado, y_va)],
            verbose=False
        )
        
        preds = xgb_model.predict(X_va_procesado)
        rmse = root_mean_squared_error(y_va, preds)
        fold_scores.append(rmse)
        
    return np.mean(fold_scores)

def run_optimization():
    print("🚀 Iniciando Ingesta para Optimización...")
    client = get_bq_client()
    df_gold = get_train_data(client)
    
    y = df_gold['log_sale_price']
    X = df_gold.drop(columns=['log_sale_price'])
    
    # El Muro de Hierro (Optuna solo ve el 85%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    print("\n🚀 Iniciando Optuna (50 Trials)...")
    # Pasamos X_train y y_train a la función objetivo usando una función lambda
    estudio = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='xgb_ames_housing'
    )
    
    estudio.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, n_jobs=1)
    
    mejor_intento = estudio.best_trial
    
    print("\n=======================================================")
    print(f"🏆 OPTIMIZACIÓN FINALIZADA")
    print(f"=======================================================")
    print(f"Mejor CV Score (RMSLE): {mejor_intento.value:.4f}")
    
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construye la ruta absoluta hacia la carpeta 'models'
    # Esto equivale a subir un nivel (..) y entrar a 'models'
    ruta_models = os.path.join(directorio_script, '..', 'models')
    ruta_json = os.path.join(ruta_models, 'best_params.json')
    
    # Asegura que la carpeta exista en la ubicación correcta
    os.makedirs(ruta_models, exist_ok=True)
    
    # Intenta guardar el archivo JSON usando la ruta absoluta fija
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(mejor_intento.params, f, indent=4, ensure_ascii=False)
        
    print("✅ Hiperparámetros guardados exitosamente en 'models/best_params.json'")

if __name__ == "__main__":
    run_optimization()