import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error

# Importaciones locales (Tu infraestructura)
from data_ingestion import get_bq_client, get_train_data, get_test_data
from features import build_preprocessor

def run_training_and_production():
    # ===================================================================
    # 1. SETUP Y LECTURA DE HIPERPARÁMETROS
    # ===================================================================
    # 1. Obtiene la carpeta exacta donde está guardado este archivo .py
    directorio_script = os.path.dirname(os.path.abspath(__file__))

    # 2. Construye la ruta absoluta hacia el archivo JSON (sube un nivel y entra a 'models')
    ruta_params = os.path.abspath(os.path.join(directorio_script, '..', 'models', 'best_params.json'))

    # 3. Verifica si el archivo existe en esa ruta absoluta fija
    if not os.path.exists(ruta_params):
        raise FileNotFoundError(f"No se encontró best_params.json en {ruta_params}. Ejecuta 'python src/optimize.py' primero.")
    with open(ruta_params, 'r') as f:
        mejores_parametros = json.load(f)

    # Agregamos reglas fijas para el Sanity Check
    params_validacion = mejores_parametros.copy()
    params_validacion['learning_rate'] = 0.05
    params_validacion['n_estimators'] = 2000 
    params_validacion['random_state'] = 42
    params_validacion['n_jobs'] = -1

    # ===================================================================
    # 2. INGESTA Y EL MURO DE HIERRO
    # ===================================================================
    client = get_bq_client()
    df_gold = get_train_data(client)
    X_test_kaggle, ids_submission = get_test_data(client)
    
    y = df_gold['log_sale_price']
    X = df_gold.drop(columns=['log_sale_price'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # ===================================================================
    # BLOQUE 7: EL SANITY CHECK (Validación Hold-Out)
    # ===================================================================
    print("⚙️ Procesando matrices para validación...")
    preprocesador_val = build_preprocessor()
    
    # Transformamos manualmente aquí solo para poder usar el eval_set de XGBoost
    X_train_procesado = preprocesador_val.fit_transform(X_train, y_train)
    X_val_procesado = preprocesador_val.transform(X_val)
    
    print("⏳ Entrenando contra el Muro de Hierro (con Early Stopping)...")
    xgb_validacion = XGBRegressor(**params_validacion, early_stopping_rounds=50)
    xgb_validacion.fit(
        X_train_procesado, y_train,
        eval_set=[(X_val_procesado, y_val)],
        verbose=False
    )
    
    preds_val = xgb_validacion.predict(X_val_procesado)
    rmse_val = root_mean_squared_error(y_val, preds_val)
    
    print("\n" + "="*55)
    print(f"🎯 SCORE FINAL (MURO DE HIERRO): {rmse_val:.4f}")
    print(f"💵 Error Comercial Aprox: ${np.expm1(rmse_val) * 180000:,.0f}")
    print("="*55)

    # ===================================================================
    # BLOQUE 8: PASE A PRODUCCIÓN Y EXPORTACIÓN
    # ===================================================================
    arboles_optimos = xgb_validacion.best_iteration
    arboles_produccion = int(arboles_optimos * 1.15) # Retrolimitación heurística
    
    print("\n🚀 INICIANDO PASE A PRODUCCIÓN...")
    print(f"Árboles óptimos en validación (85% datos): {arboles_optimos}")
    print(f"Árboles asignados para producción (100% datos): {arboles_produccion}")
    
    # Configuramos el modelo final ciego (sin early stopping)
    params_prod = mejores_parametros.copy()
    params_prod.update({
        'learning_rate': 0.05,
        'n_estimators': arboles_produccion,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # EL SECRETO MLOps: Unimos TODO en una sola tubería
    pipeline_produccion = Pipeline(steps=[
        ('preprocesador_maestro', build_preprocessor()),
        ('algoritmo_predictivo', XGBRegressor(**params_prod))
    ])
    
    print("⚙️ Entrenando la tubería definitiva con el 100% de los datos históricos...")
    pipeline_produccion.fit(X, y)
    print("✅ Entrenamiento ciego completado.")
    
    # 1. EXPORTACIÓN DEL ARTEFACTO (.joblib)
    
    ruta_models = os.path.join(directorio_script, '..', 'models')
    os.makedirs(ruta_models, exist_ok=True)
    ruta_modelo = os.path.join(ruta_models,'pipeline_produccion_v1.joblib')
    joblib.dump(pipeline_produccion, ruta_modelo)
    print(f"💾 Tubería de producción exportada exitosamente a: {ruta_modelo}")
    
    # 2. GENERACIÓN DEL SUBMISSION DE KAGGLE (Inferencia)
    print("🧠 Realizando predicciones sobre casas invisibles (Kaggle Test)...")
    # Al usar el pipeline completo, solo llamamos a .predict(), 
    # él se encarga de limpiar, podar y predecir internamente.
    predicciones_log = pipeline_produccion.predict(X_test_kaggle)
    predicciones_dolares = np.expm1(predicciones_log)
    
    df_submission = pd.DataFrame({
        'Id': ids_submission,
        'SalePrice': predicciones_dolares
    })
    
    nombre_archivo = 'submission_xgboost_pro.csv'
    df_submission.to_csv(nombre_archivo, index=False)
    
    print("\n" + "="*55)
    print(f"🎉 ¡SISTEMA END-TO-END COMPLETADO!")
    print(f"Archivo '{nombre_archivo}' generado y listo para Kaggle.")
    print("="*55)

if __name__ == "__main__":
    run_training_and_production()