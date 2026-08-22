import os
from dotenv import load_dotenv

# 1. Cargar las variables de entorno desde el archivo .env oculto
load_dotenv()

# 2. Constantes globales del sistema
# Centralizamos el random_state acá para asegurar reproducibilidad 
# en todo el pipeline (splits, modelos, K-Means, etc.)
RANDOM_STATE = 42

# 3. Validación de seguridad temprana
# Si la credencial no está en el entorno, el sistema avisa antes de que algo explote
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    print("⚠️ ADVERTENCIA: La variable GOOGLE_APPLICATION_CREDENTIALS no fue encontrada en el entorno.")