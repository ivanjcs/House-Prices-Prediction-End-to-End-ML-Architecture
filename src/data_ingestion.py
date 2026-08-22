import os
import pandas as pd
from google.cloud import bigquery

# Importamos config para que al ejecutarse este script, 
# se carguen las variables de entorno automáticamente.
import config 

def get_bq_client():
    """
    Inicializa y retorna el cliente de BigQuery solo si estamos en modo PROD.
    Retorna None en modo LOCAL para evitar errores de autenticación.
    """
    modo = os.getenv("EXECUTION_MODE", "LOCAL")
    
    if modo == "LOCAL":
        return None
        
    try:
        client = bigquery.Client()
        print("✅ Cliente de BigQuery inicializado correctamente (Modo PROD).")
        return client
    except Exception as e:
        print(f"❌ Error al conectar con BigQuery: {e}")
        raise

def get_train_data(client: bigquery.Client = None) -> pd.DataFrame:
    """Descarga los datos de entrenamiento dependiendo del entorno."""
    modo = os.getenv("EXECUTION_MODE", "LOCAL")
    
    if modo == "PROD":
        # Leemos el ID del proyecto desde el .env
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("❌ Falta definir GCP_PROJECT_ID en el archivo .env")

        print(f"☁️ [PROD] Descargando datos de entrenamiento (Train) desde {project_id}...")
        
        # Inyectamos el project_id dinámicamente en la query usando una f-string
        query = f"""
            SELECT *
            FROM `{project_id}.dbt_icastro_gold_marts.obt_house_prices__train`
        """
        df_train = client.query(query).to_dataframe()
    else:
        print("💻 [LOCAL] Leyendo datos de entrenamiento desde CSV local...")
        
        # 1. Obtiene la ruta del script actual
        directorio_script = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construye la ruta absoluta hacia la carpeta 'data' subiendo un nivel
        ruta_data = os.path.join(directorio_script, '..', 'data')
        ruta_csv = os.path.join(ruta_data, 'obt_house_prices__train.csv')
        
        # 3. Verifica si el archivo existe usando la ruta absoluta definitiva
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"Falta el archivo {ruta_csv}. Descárgalo de BQ y ponlo en la carpeta data/.")
            
        # 4. Lee el archivo de forma segura
        df_train = pd.read_csv(ruta_csv)
    
    # Limpieza básica de la ingesta
    if 'property_id' in df_train.columns:
        df_train = df_train.drop(columns=['property_id'])
        
    print(f"✅ Datos de Train listos. Filas: {df_train.shape[0]}, Columnas: {df_train.shape[1]}")
    return df_train

def get_test_data(client: bigquery.Client = None) -> tuple[pd.DataFrame, pd.Series]:
    """Descarga los datos de testeo y separa los IDs dependiendo del entorno."""
    modo = os.getenv("EXECUTION_MODE", "LOCAL")
    
    if modo == "PROD":
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("❌ Falta definir GCP_PROJECT_ID en el archivo .env")

        print(f"☁️ [PROD] Descargando datos de test (Kaggle Test) desde {project_id}...")
        
        query_test = f"""
            SELECT *
            FROM `{project_id}.dbt_icastro_gold_marts.obt_house_prices__test`
        """
        df_test = client.query(query_test).to_dataframe()
    else:
        print("💻 [LOCAL] Leyendo datos de testing desde CSV local...")
        
        # 1. Obtiene la ruta del script actual
        directorio_script = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construye la ruta absoluta hacia la carpeta 'data' subiendo un nivel
        ruta_data = os.path.join(directorio_script, '..', 'data')
        ruta_csv = os.path.join(ruta_data, 'obt_house_prices__test.csv')
        
        # 3. Verifica si el archivo existe usando la ruta absoluta definitiva
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"Falta el archivo {ruta_csv}. Descárgalo de BQ y ponlo en la carpeta data/.")
            
        # 4. Lee el archivo de forma segura
        df_test = pd.read_csv(ruta_csv)
    
    # Validación de datos y auditoría básica
    duplicados = df_test[df_test.duplicated(subset=['property_id'], keep=False)]
    if not duplicados.empty:
        print("🚨 ADVERTENCIA: Se encontraron casas duplicadas en Test.")
        print(duplicados[['property_id', 'Neighborhood', 'SaleCondition']])

    # Separar IDs y limpiar
    ids_submission = df_test['property_id']
    X_test = df_test.drop(columns=['property_id', 'Id'], errors='ignore')
    
    print(f"✅ Datos de Test listos. Filas: {X_test.shape[0]}")
    return X_test, ids_submission

# Este bloque permite probar el script individualmente 
if __name__ == "__main__":
    bq_client = get_bq_client()
    df_entrenamiento = get_train_data(bq_client)
    df_prueba, ids_prueba = get_test_data(bq_client)