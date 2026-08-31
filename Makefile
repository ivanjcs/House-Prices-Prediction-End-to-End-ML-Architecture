.PHONY: run-batch demo-local

# Ejecuta el flujo completo de producción en la nube
run-batch:
	@echo "Iniciando tubería de procesamiento Batch (Producción)..."
	@echo "Paso 1: Transformación masiva (dbt)..."
	dbt run --select stg_house_prices__new obt_house_prices__to_predict
	@echo "Paso 2: Inferencia de Machine Learning (Python)..."
	poetry run python src/batch_inference.py
	@echo "✅ Pipeline orquestado exitosamente."

# Ejecuta solo la simulación local para portafolio
demo-local:
	@echo "Iniciando simulación local de tasaciones..."
	poetry run python src/batch_inference.py
	@echo "✅ Demo finalizada."