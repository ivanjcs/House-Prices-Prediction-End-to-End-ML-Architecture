-- models/staging/stg_int_house_prices_modes.sql

{% set columnas_moda = [
    ('Electrical', 'electrical'), ('MSZoning', 'ms_zoning'), ('Utilities', 'utilities'), 
    ('Exterior1st', 'exterior1st'), ('Exterior2nd', 'exterior2nd'), ('KitchenQual', 'kitchen_qual'), 
    ('Functional', 'functional'), ('SaleType', 'sale_type')
] %}

WITH raw_train AS (
    -- REGLA DE ORO: Solo aprendemos las modas del conjunto histórico (Train)
    SELECT * 
    FROM {{ source('bronze_raw', 'house_prices_train') }}
),

global_modes AS (
    SELECT
        {% for col_src, col_alias in columnas_moda %}
        (
            SELECT {{ col_src }} 
            FROM raw_train 
            WHERE {{ col_src }} IS NOT NULL AND CAST({{ col_src }} AS STRING) != 'NA'
            GROUP BY {{ col_src }} 
            ORDER BY COUNT(*) DESC 
            LIMIT 1
        ) AS mode_{{ col_alias }}{% if not loop.last %},{% endif %}
        {% endfor %}
)

-- El resultado de esto es una tabla de EXACTAMENTE 1 fila y 8 columnas
SELECT * FROM global_modes