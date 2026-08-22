import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Importamos nuestros módulos locales
from data_ingestion import get_bq_client, get_train_data
from features import build_preprocessor

def evaluate_baselines():
    print("🚀 Iniciando evaluación de modelos Baseline...")
    
    # 1. Ingesta de datos
    client = get_bq_client()
    df_gold = get_train_data(client)
    
    # 2. El Muro de Hierro (Aseguramos evaluar sobre los mismos datos)
    y = df_gold['log_sale_price']
    X = df_gold.drop(columns=['log_sale_price'])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # 3. Instanciar la fábrica de variables (El pipeline completo)
    pipeline_maestro = build_preprocessor()
    
    # 4. Definición de modelos
    modelos_baseline = {
        "Ridge Regression (Lineal)": Ridge(random_state=42),
        "XGBoost Default (Árboles)": XGBRegressor(random_state=42, n_jobs=-1)
    }
    
    # 5. Evaluación Competitiva
    print("\n⏳ Entrenando Modelos Baseline con 5-Fold Cross-Validation...")
    print("💡 NOTA: El RMSE devuelto ES el RMSLE de Kaggle.\n")
    
    resultados_rmse = {}
    
    for nombre_modelo, modelo in modelos_baseline.items():
        # Enchufamos la tubería COMPLETA al algoritmo
        pipeline_actual = Pipeline(steps=[
            ('preprocesamiento_completo', pipeline_maestro), 
            ('algoritmo', modelo)
        ])
        
        scores = cross_val_score(
            pipeline_actual, 
            X_train, 
            y_train, 
            cv=5, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        rmse_scores = np.abs(scores)
        resultados_rmse[nombre_modelo] = rmse_scores.mean()
        
        print("=======================================================")
        print(f"📊 Reporte detallado para: {nombre_modelo}")
        print("=======================================================")
        for i, fold_score in enumerate(rmse_scores):
            print(f"   -> Pliegue {i+1}: RMSLE = {fold_score:.4f}")
        
        print("   ----------------------------------------------------")
        print(f"✅ {nombre_modelo}")
        print(f"   -> Kaggle Score (RMSLE): {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
        
        # Cálculo de negocio
        porcentaje_error = np.expm1(rmse_scores.mean())
        error_dolares_aprox = 180000 * porcentaje_error 
        print(f"   -> Error aprox. en dólares: ${error_dolares_aprox:,.0f}\n")

    # 6. Veredicto
    mejor_modelo = min(resultados_rmse, key=resultados_rmse.get)
    print(f"🏆 El ganador del Baseline es: {mejor_modelo}")

if __name__ == "__main__":
    evaluate_baselines()