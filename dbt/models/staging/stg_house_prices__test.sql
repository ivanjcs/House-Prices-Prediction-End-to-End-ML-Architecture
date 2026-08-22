{% set columnas_na = [
    ('PoolQC', 'pool_qc'), ('MiscFeature', 'misc_feature'), ('Alley', 'alley'), 
    ('Fence', 'fence'), ('FireplaceQu', 'fireplace_qu'), ('GarageType', 'garage_type'), 
    ('GarageFinish', 'garage_finish'), ('GarageQual', 'garage_qual'), ('GarageCond', 'garage_cond'), 
    ('BsmtCond', 'bsmt_cond'), ('BsmtExposure', 'bsmt_exposure'), ('BsmtQual', 'bsmt_qual'), 
    ('BsmtFinType1', 'bsmt_fin_type1'), ('BsmtFinType2', 'bsmt_fin_type2'), ('MasVnrType', 'mas_vnr_type')
] %}

-- Creamos la lista para el Grupo "Cero Lógico"
{% set columnas_zero = [
    ('GarageCars', 'garage_cars'), ('GarageArea', 'garage_area'), ('BsmtFullBath', 'bsmt_full_bath'), 
    ('BsmtHalfBath', 'bsmt_half_bath'), ('BsmtUnfSF', 'bsmt_unf_sf'), ('BsmtFinSF1', 'bsmt_fin_sf1'), 
    ('BsmtFinSF2', 'bsmt_fin_sf2'), ('TotalBsmtSF', 'total_bsmt_sf'), ('MasVnrArea', 'mas_vnr_area'), 
    ('GarageYrBlt', 'garage_yr_blt')
] %}

-- Grupo "Moda General" para variables categóricas con nulos esporádicos
{% set columnas_moda = [
    ('Electrical', 'electrical'), ('MSZoning', 'ms_zoning'), ('Utilities', 'utilities'), 
    ('Exterior1st', 'exterior1st'), ('Exterior2nd', 'exterior2nd'), ('KitchenQual', 'kitchen_qual'), 
    ('Functional', 'functional'), ('SaleType', 'sale_type')
] %}

WITH raw_source AS (
    SELECT * FROM {{ source('bronze_raw', 'house_prices_test') }}
),

-- Calculamos la Moda General (el valor más frecuente ignorando nulos) para cada columna
historical_modes AS (
    SELECT *
    FROM {{ ref('stg_int_house_prices_modes') }}
),

renamed AS (
    SELECT 
        -- 1. Columnas sin modificación (LIMPIO DE DUPLICADOS)s.Id AS property_id,
        s.Id AS property_id,
        s.MSSubClass AS mssub_class,
        s.LotArea AS lot_area,
        s.Street AS street,
        s.LotShape AS lot_shape,
        s.LandContour AS land_contour,
        s.LotConfig AS lot_config,
        s.LandSlope AS land_slope,
        s.Neighborhood AS neighborhood,
        s.Condition1 AS condition1,
        s.Condition2 AS condition2,
        s.BldgType AS bldg_type,
        s.HouseStyle AS house_style,
        s.OverallQual AS overall_qual,
        s.OverallCond AS overall_cond,
        s.YearBuilt AS year_built,
        s.YearRemodAdd AS year_remod_add,
        s.RoofStyle AS roof_style,
        s.RoofMatl AS roof_matl,
        s.ExterQual AS exter_qual,
        s.ExterCond AS exter_cond,
        s.Foundation AS foundation,
        s.Heating AS heating,
        s.HeatingQC AS heating_qc,
        s.CentralAir AS central_air,
        s.LowQualFinSF AS low_qual_fin_sf,
        s.GrLivArea AS gr_liv_area,
        s.FullBath AS full_bath,
        s.HalfBath AS half_bath,
        s.BedroomAbvGr AS bedroom_abv_gr,
        s.KitchenAbvGr AS kitchen_abv_gr,
        s.TotRmsAbvGrd AS tot_rms_abv_grd,
        s.Fireplaces AS fireplaces,
        s.PavedDrive AS paved_drive,
        s.WoodDeckSF AS wood_deck_sf,
        s.OpenPorchSF AS open_porch_sf,
        s.EnclosedPorch AS enclosed_porch,
        s.ScreenPorch AS screen_porch,
        s.PoolArea AS pool_area,
        s.MiscVal AS misc_val,
        s.MoSold AS mo_sold,
        s.YrSold AS yr_sold,
        s.SaleCondition AS sale_condition,
        --s.SalePrice AS sale_price,

        -- ALERTA: Busca las filas de los pisos y cámbialas manualmente para que empiecen con letras:
        `1stFlrSF` AS first_flr_sf,
        `2ndFlrSF` AS second_flr_sf,
        `3SsnPorch` AS three_ssn_porch,

        -- 2. Imputación Avanzada: lot_frontage (Mediana por Barrio)
        COALESCE(
            SAFE_CAST(s.LotFrontage AS FLOAT64),
            PERCENTILE_CONT(SAFE_CAST(s.LotFrontage AS FLOAT64), 0.5) OVER (PARTITION BY s.Neighborhood)
        ) AS lot_frontage,

        -- 3 . Imputación por Moda General Histórica
        {% for col_src, col_alias in columnas_moda %}
        COALESCE(NULLIF(CAST(s.{{ col_src }} AS STRING), 'NA'), m.mode_{{ col_alias }}) AS {{ col_alias }},
        {% endfor %}

        -- 4. Grupo "El Vacío es una Categoría"
        {% for col_src, col_alias in columnas_na %}
        COALESCE(NULLIF(CAST(s.{{ col_src }} AS STRING), 'NA'), 'None') AS {{ col_alias }},
        {% endfor %}

        -- 5. Grupo "El Vacío es un Cero Lógico"
        {% for col_src, col_alias in columnas_zero %}
        COALESCE(SAFE_CAST(s.{{ col_src }} AS INT64), 0) AS {{ col_alias }}{% if not loop.last %},{% endif %}
        {% endfor %}

    FROM raw_source AS s
    CROSS JOIN historical_modes AS m
)

SELECT * FROM renamed