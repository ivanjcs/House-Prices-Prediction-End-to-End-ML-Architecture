import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import bigquery

# Importamos tu módulo de ingesta
from data_ingestion import get_bq_client, get_to_predict_data

load_dotenv()

def ejecutar_tasacion_masiva():
    modo = os.getenv("EXECUTION_MODE", "LOCAL")
    print(f"🚀 Iniciando Motor de Tasación Batch (Modo: {modo})")
    print("-" * 50)
    
    # --- CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # Ruta absoluta para buscar el modelo
    ruta_models = os.path.join(directorio_script, '..', 'models')
    ruta_modelo = os.path.join(ruta_models, 'pipeline_produccion_v1.joblib')
    
    # Ruta absoluta para guardar la salida local
    ruta_data = os.path.join(directorio_script, '..', 'data')
    ruta_salida = os.path.join(ruta_data, 'predicciones.csv')
    # ----------------------------------------
    
    # 1. Extracción (Agnóstica al entorno)
    client = get_bq_client()
    X_predict, ids = get_to_predict_data(client)
    
    if X_predict.empty:
        print("✅ No hay propiedades nuevas en la cola para procesar.")
        return

    # 2. Inferencia (El Cerebro)
    print("🧠 Cargando Pipeline de Machine Learning...")
    pipeline = joblib.load(ruta_modelo)
    
    print(f"⏳ Ejecutando inferencia para {len(X_predict)} propiedades...")
    predicciones_log = pipeline.predict(X_predict)
    
    # Deshacemos el logaritmo
    precios_dolares = np.expm1(predicciones_log)
    
    # Preparamos el DataFrame de salida
    df_resultados = pd.DataFrame({
        'property_id': ids,
        'predicted_price_usd': precios_dolares,
        'execution_date': pd.Timestamp.now() # Fecha de tasación para auditoría
    })
    
    # 3. Carga / Write-back (El Guardado Inteligente)
    print("-" * 50)
    if modo == "PROD":
        project_id = os.getenv("GCP_PROJECT_ID")
        tabla_destino = f"{project_id}.dbt_icastro_gold_marts.fct_predicciones_inmobiliarias"
        
        print(f"☁️ [PROD] Subiendo resultados a BigQuery: {tabla_destino}...")
        
        
        # Usamos el cliente nativo de Google Cloud en lugar de Pandas-GBQ
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND", # Añadimos a la tabla si ya existe
        )
        
        # Ejecutamos el trabajo de carga
        job = client.load_table_from_dataframe(
            df_resultados, 
            tabla_destino, 
            job_config=job_config
        )
        
        # Esperamos a que BigQuery confirme que terminó
        job.result()
        print("✅ Tasaciones guardadas exitosamente en la capa Platinum (Data Warehouse).")
    else:
        # Asegura que la carpeta exista antes de guardar el CSV local
        os.makedirs(ruta_data, exist_ok=True)
        
        df_resultados.to_csv(ruta_salida, index=False)
        print(f"💻 [LOCAL] Resultados guardados localmente en: {ruta_salida}")
        print("\n📊 Vista previa de las tasaciones:")
        print(df_resultados.head().to_string(index=False, float_format=lambda x: f"${x:,.2f}"))

if __name__ == "__main__":
    ejecutar_tasacion_masiva()
