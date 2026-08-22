-- models/marts/obt_house_prices__to_predict.sql

WITH new_data AS (
    SELECT * FROM {{ ref('stg_house_prices__new') }}
),

encoded_data AS (
    SELECT
        -- 1. Exclusión de variables crudas + Las 2 que vamos a "disfrazar"
        * EXCEPT(
            first_flr_sf, second_flr_sf, garage_area, garage_yr_blt,
            pool_area, mas_vnr_area, year_built, year_remod_add,
            wood_deck_sf, open_porch_sf, enclosed_porch, three_ssn_porch, screen_porch,
            full_bath, half_bath, bsmt_full_bath, bsmt_half_bath,
            condition1, condition2, mo_sold,

            -- [Exclusiones definidas por la Auditoria Visual] --------------

            -- Exclusiones por varianza cercana a cero
            low_qual_fin_sf, misc_val,

            -- Exclusiones por imbalances extremos sin poder predictivo
            street, utilities, pool_qc,

            -- Exclusiones por Alta correlación (Multicolinealidad)
            bsmt_cond, bsmt_fin_type1, exterior2nd, garage_cond, 
            garage_qual, mssub_class, tot_rms_abv_grd, ms_zoning,

            -- Exclusion de variables crudas que vamos a binarizar/agrupar ahora
            roof_matl, heating, central_air, kitchen_abv_gr, electrical, misc_feature,

            exter_qual, exter_cond, bsmt_qual, heating_qc, kitchen_qual, fireplace_qu, 
            functional, bsmt_fin_type2,
            bsmt_exposure, lot_shape, garage_finish, paved_drive, land_slope, alley, bsmt_unf_sf
        ),
        -- [ Feature Engineering ] ------------
        COALESCE((bsmt_fin_sf1 + bsmt_fin_sf2) / NULLIF(total_bsmt_sf, 0), 0) AS porcentaje_sotano_terminado,
        SAFE_DIVIDE(gr_liv_area, lot_area) AS lot_coverage_pct,
        CASE WHEN garage_cars = 0 THEN -999 ELSE (yr_sold - garage_yr_blt) END AS garage_age,
        CASE WHEN pool_area > 0 THEN 1 ELSE 0 END AS has_pool,
        CASE WHEN mas_vnr_area > 0 THEN 1 ELSE 0 END AS has_masonry,
        (yr_sold - year_built) AS house_age,
        (yr_sold - year_remod_add) AS years_since_remodel,
        CASE WHEN year_built != year_remod_add THEN 1 ELSE 0 END AS has_been_remodeled,

        (wood_deck_sf + open_porch_sf + enclosed_porch + three_ssn_porch + screen_porch) AS total_outdoor_space,

        CASE 
            WHEN (wood_deck_sf + open_porch_sf + enclosed_porch + three_ssn_porch + screen_porch) > 0 THEN 1 
            ELSE 0 
        END AS has_outdoor_space,

        (full_bath + (0.5 * half_bath) + bsmt_full_bath + (0.5 * bsmt_half_bath)) AS total_bathrooms,

        CASE 
            WHEN condition1 IN ('Artery', 'Feedr', 'RRNn', 'RRAn', 'RRNe', 'RRAe') 
              OR condition2 IN ('Artery', 'Feedr', 'RRNn', 'RRAn', 'RRNe', 'RRAe') THEN 1 
            ELSE 0 
        END AS is_noisy,

        CASE 
            WHEN condition1 IN ('PosN', 'PosA') 
              OR condition2 IN ('PosA', 'PosN') THEN 1 
            ELSE 0 
        END AS is_near_park,

        SIN(2 * 3.141592653589793 * mo_sold / 12) AS mo_sold_sin,
        COS(2 * 3.141592653589793 * mo_sold / 12) AS mo_sold_cos,
        CASE WHEN mo_sold BETWEEN 4 AND 7 THEN 1 ELSE 0 END AS es_temporada_alta,

        (enclosed_porch + three_ssn_porch + screen_porch) AS total_living_porch,
        (wood_deck_sf + open_porch_sf) AS total_outdoor_deck,


        -- [ REGLAS DEFINIDAS por (Auditoria Visual )] ---------------------------

        -- Imbalance: Binarización
        -- Si es el material estándar (CompShg) es 1, cualquier otra rareza es 0
        CASE WHEN roof_matl = 'CompShg' THEN 1 ELSE 0 END AS has_standard_roof,

        -- Si es calefacción a gas estándar (GasA) es 1, otro es 0
        CASE WHEN heating = 'GasA' THEN 1 ELSE 0 END AS has_gas_heating,

        -- CentralAir ya es Y/N, lo pasamos a 1/0
        CASE WHEN CAST(central_air AS STRING) IN ('Y', 'true', 'TRUE') THEN 1 ELSE 0 END AS has_central_air,

        -- Cocinas: 1 es lo normal, más de 1 es raro/multifamiliar
        CASE WHEN kitchen_abv_gr = 1 THEN 1 ELSE 0 END AS is_single_kitchen,

        -- Imbalance: Agrupación (Binning)

        -- Sistema eléctrico: 1 si es estándar (SBrkr), 0 si es antiguo/peligroso
        CASE WHEN electrical = 'SBrkr' THEN 1 ELSE 0 END AS has_standard_electrical,    

        -- MiscFeature: Separamos las casas que tienen un rasgo raro (Shed, etc) de las que no
        -- Ojo aquí con los valores nulos que vienen de la capa bronce
        CASE WHEN COALESCE(misc_feature, 'None') = 'None' THEN 0 ELSE 1 END AS has_misc_feature,

        -- [ ORDINAL ENCODING (Escalas de valor) ] --------------------------------
        -- 1. Escala Estándar de Calidad y Condición
        CASE WHEN exter_qual = 'Ex' THEN 5 WHEN exter_qual = 'Gd' THEN 4 WHEN exter_qual = 'TA' THEN 3 WHEN exter_qual = 'Fa' THEN 2 WHEN exter_qual = 'Po' THEN 1 ELSE 0 END AS exter_qual,
        CASE WHEN exter_cond = 'Ex' THEN 5 WHEN exter_cond = 'Gd' THEN 4 WHEN exter_cond = 'TA' THEN 3 WHEN exter_cond = 'Fa' THEN 2 WHEN exter_cond = 'Po' THEN 1 ELSE 0 END AS exter_cond,
        CASE WHEN bsmt_qual = 'Ex' THEN 5 WHEN bsmt_qual = 'Gd' THEN 4 WHEN bsmt_qual = 'TA' THEN 3 WHEN bsmt_qual = 'Fa' THEN 2 WHEN bsmt_qual = 'Po' THEN 1 ELSE 0 END AS bsmt_qual,
        CASE WHEN heating_qc = 'Ex' THEN 5 WHEN heating_qc = 'Gd' THEN 4 WHEN heating_qc = 'TA' THEN 3 WHEN heating_qc = 'Fa' THEN 2 WHEN heating_qc = 'Po' THEN 1 ELSE 0 END AS heating_qc,
        CASE WHEN kitchen_qual = 'Ex' THEN 5 WHEN kitchen_qual = 'Gd' THEN 4 WHEN kitchen_qual = 'TA' THEN 3 WHEN kitchen_qual = 'Fa' THEN 2 WHEN kitchen_qual = 'Po' THEN 1 ELSE 0 END AS kitchen_qual,
        CASE WHEN fireplace_qu = 'Ex' THEN 5 WHEN fireplace_qu = 'Gd' THEN 4 WHEN fireplace_qu = 'TA' THEN 3 WHEN fireplace_qu = 'Fa' THEN 2 WHEN fireplace_qu = 'Po' THEN 1 ELSE 0 END AS fireplace_qu,

        -- (Opcional) Si decidimos revivir las variables correlacionadas, descomenta estas líneas:
        -- CASE WHEN bsmt_cond = 'Ex' THEN 5 WHEN bsmt_cond = 'Gd' THEN 4 WHEN bsmt_cond = 'TA' THEN 3 WHEN bsmt_cond = 'Fa' THEN 2 WHEN bsmt_cond = 'Po' THEN 1 ELSE 0 END AS bsmt_cond,
        -- CASE WHEN garage_qual = 'Ex' THEN 5 WHEN garage_qual = 'Gd' THEN 4 WHEN garage_qual = 'TA' THEN 3 WHEN garage_qual = 'Fa' THEN 2 WHEN garage_qual = 'Po' THEN 1 ELSE 0 END AS garage_qual,
        -- CASE WHEN garage_cond = 'Ex' THEN 5 WHEN garage_cond = 'Gd' THEN 4 WHEN garage_cond = 'TA' THEN 3 WHEN garage_cond = 'Fa' THEN 2 WHEN garage_cond = 'Po' THEN 1 ELSE 0 END AS garage_cond,

        -- 2. Escala de Funcionalidad
        CASE 
            WHEN functional = 'Typ' THEN 7 WHEN functional = 'Min1' THEN 6 WHEN functional = 'Min2' THEN 5 
            WHEN functional = 'Mod' THEN 4 WHEN functional = 'Maj1' THEN 3 WHEN functional = 'Maj2' THEN 2 
            WHEN functional = 'Sev' THEN 1 WHEN functional = 'Sal' THEN 0 
            ELSE 7 -- Asumimos típica (Typ) por defecto si hay anomalías
        END AS functional,

        -- 3. Acabados del Sótano
        CASE WHEN bsmt_fin_type2 = 'GLQ' THEN 6 WHEN bsmt_fin_type2 = 'ALQ' THEN 5 WHEN bsmt_fin_type2 = 'BLQ' THEN 4 WHEN bsmt_fin_type2 = 'Rec' THEN 3 WHEN bsmt_fin_type2 = 'LwQ' THEN 2 WHEN bsmt_fin_type2 = 'Unf' THEN 1 ELSE 0 END AS bsmt_fin_type2,

        -- (Opcional) Si decidimos revivir bsmt_fin_type1:
        -- CASE WHEN bsmt_fin_type1 = 'GLQ' THEN 6 WHEN bsmt_fin_type1 = 'ALQ' THEN 5 WHEN bsmt_fin_type1 = 'BLQ' THEN 4 WHEN bsmt_fin_type1 = 'Rec' THEN 3 WHEN bsmt_fin_type1 = 'LwQ' THEN 2 WHEN bsmt_fin_type1 = 'Unf' THEN 1 ELSE 0 END AS bsmt_fin_type1,

        -- 4. Exposición del sótano
        CASE WHEN bsmt_exposure = 'Gd' THEN 4 WHEN bsmt_exposure = 'Av' THEN 3 WHEN bsmt_exposure = 'Mn' THEN 2 WHEN bsmt_exposure = 'No' THEN 1 ELSE 0 END AS bsmt_exposure,

        -- 5. Irregularidad del terreno
        CASE WHEN lot_shape = 'Reg' THEN 3 WHEN lot_shape = 'IR1' THEN 2 WHEN lot_shape = 'IR2' THEN 1 WHEN lot_shape = 'IR3' THEN 0 ELSE 3 END AS lot_shape,

        -- 6. Interior del garage
        CASE WHEN garage_finish = 'Fin' THEN 3 WHEN garage_finish = 'RFn' THEN 2 WHEN garage_finish = 'Unf' THEN 1 ELSE 0 END AS garage_finish,

        -- 7. Camino de entrada pavimentado
        CASE WHEN paved_drive = 'Y' THEN 2 WHEN paved_drive = 'P' THEN 1 WHEN paved_drive = 'N' THEN 0 ELSE 0 END AS paved_drive,

        -- 8. Inclinación del terreno
        CASE WHEN land_slope = 'Gtl' THEN 2 WHEN land_slope = 'Mod' THEN 1 WHEN land_slope = 'Sev' THEN 0 ELSE 2 END AS land_slope,

        -- 9. Callejón (Alley)
        CASE WHEN alley = 'Pave' THEN 2 WHEN alley = 'Grvl' THEN 1 ELSE 0 END AS alley

        -- NOTA CRUCIAL: No se incluye LOG(sale_price) porque test.csv no tiene precio.

    FROM new_data
),

engineered_data AS (
    SELECT 
        * EXCEPT(
            has_pool,
            is_near_park,
            has_misc_feature,
            has_masonry
        ),
        -- Traemos todo lo del bloque anterior (donde exter_cond ya es un número INT64)
        
        -- Índice de Deterioro Continuo (Ahora la matemática funcionará perfecto)
        ((5 - exter_cond) + (5 - heating_qc)) AS deterioration_index,
    
        -- Bandera Crítica de Reparación Urgente 
        CASE WHEN exter_cond <= 2 AND heating_qc <= 3 THEN 1 ELSE 0 END AS needs_urgent_repair,

                -- =========================================================================
        -- HIPÓTESIS A: EL CASTIGO POR OBSOLESCENCIA Y RIESGO (SHAP INSIGHTS)
        -- =========================================================================

        -- 1. Edad Efectiva
        -- Aísla la cantidad de años que la estructura lleva envejeciendo de forma real
        -- sin recibir una inyección de capital o renovación profunda.
        (house_age - years_since_remodel) AS edad_efectiva,

        -- 2. Propiedad de Riesgo (Flag de Dealbreakers)
        -- Fuerza al algoritmo a aislar en una rama de alta penalización a las casas
        -- que combinan baja calidad constructiva en zonas de bajo valor o que carecen de pavimento.
        --CASE 
        --    WHEN (overall_qual < 5 AND neighborhood = 0) OR paved_drive = 0 
        --    THEN 1 
        --    ELSE 0 
        --END AS es_propiedad_riesgo,

        
        -- 3. Penalización por Falsa Condición (Anti "Lipstick on a Pig")
        -- Identifica la contradicción detectada en SHAP: propiedades con un estado de conservación 
        -- aceptable o alto (overall_cond >= 6) pero cuya estructura base está obsoleta (edad_efectiva > 40) 
        -- o es de materiales deficientes (overall_qual < 5). Retorna un Flag binario de advertencia.
        CASE 
            WHEN overall_cond >= 6 AND (overall_qual < 5 OR (house_age - years_since_remodel) > 40) 
            THEN 1 
            ELSE 0 
        END AS penalizacion_condicion

        -- =========================================================================
        -- HIPÓTESIS B: PROPORCIONES ARQUITECTÓNICAS (SHAP INSIGHTS)
        -- =========================================================================

        -- Ratio Sótano / Área Habitable
        -- Un valor cercano a 1.0 mapea viviendas de una sola planta perfectamente simétricas (Ranch).
        -- SAFE_DIVIDE retorna NULL en lugar de romper el pipeline si gr_liv_area fuese 0.
        -- COALESCE reemplaza ese NULL por 0 para que XGBoost reciba un valor numérico limpio.
        --COALESCE(SAFE_DIVIDE(total_bsmt_sf, gr_liv_area), 0) AS ratio_sotano_habitable
        
    FROM encoded_data
)
SELECT *
FROM engineered_data