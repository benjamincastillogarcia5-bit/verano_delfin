import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def preprocesar_rndc_grupo1(ruta_archivo):
    """
    Pipeline de limpieza (Semana 1) para el análisis de sesgo territorial
    en el transporte de carga colombiano, enfocado en la "ceguera algorítmica
    intermodal": la tendencia de los modelos a penalizar rutas que pasan por
    nodos intermodales (La Dorada, Chiriguaná, Puerto Salgar, Cartagena).

    Fuente de datos: RNDC (Registro Nacional de Despachos de Carga),
    administrado por la Superintendencia de Transporte (SuperTransporte).
    Cada fila representa un despacho de carga registrado obligatoriamente
    por las empresas transportadoras bajo el Decreto 1017 / SICE-TAC.

    Retorna: DataFrame limpio listo para modelamiento (Semana 2).
    """
    print("=== INICIANDO PROTOCOLO DE LIMPIEZA RNDC - GRUPO 1 ===")

    # =========================================================================
    # PASO 1 — CARGA DEL DATASET
    # =========================================================================
    # Se carga el CSV de microdatos del RNDC.
    # - sep=',' y quotechar='"' son los delimitadores estándar del RNDC.
    # - low_memory=False obliga a pandas a inferir el tipo de cada columna
    #   leyendo el archivo completo de una vez. Sin esto, pandas lee por
    #   bloques y puede asignar tipos inconsistentes (e.g., una columna que
    #   es float en el bloque 1 y str en el bloque 2), generando warnings
    #   y conversiones silenciosas erróneas.
    try:
        df = pd.read_csv(ruta_archivo, sep=',', quotechar='"', low_memory=False)
        print(f"Datos originales cargados: {df.shape[0]} registros.")
    except Exception as e:
        return f"Error al cargar el archivo: {e}"

    # =========================================================================
    # PASO 2 — CORRECCIÓN TEMPORAL: MITIGACIÓN DEL DESFASE DE 1899
    # =========================================================================
    # PROBLEMA ORIGINAL EN EL RNDC:
    # El campo 'hora_salida_cargue' proviene de una exportación de Excel.
    # Excel representa las horas como fracciones decimales de un día contado
    # desde el 30 de diciembre de 1899 (su fecha base interna). Al importar
    # ese campo directamente, pandas lo interpreta como un timestamp del
    # siglo XIX (ej: "1899-12-30 14:30:00"), haciendo inútil cualquier
    # cálculo de duración o ventana temporal.
    #
    # SOLUCIÓN:
    # Se extraen por separado la parte de FECHA (de 'fechasalidacargue',
    # que sí tiene el año correcto) y la parte de HORA (de 'hora_salida_cargue',
    # que tiene la hora correcta pero el año incorrecto), y luego se combinan
    # en un único timestamp válido: FECHA_HORA_SALIDA_CORREGIDA.
    # errors='coerce' convierte cualquier valor que no pueda parsearse en NaT
    # (Not a Time), evitando que un registro malo rompa todo el pipeline.
    if 'fechasalidacargue' in df.columns and 'hora_salida_cargue' in df.columns:
        fecha_real = pd.to_datetime(df['fechasalidacargue'], errors='coerce').dt.date
        hora_real  = pd.to_datetime(df['hora_salida_cargue'], errors='coerce').dt.time
        df['FECHA_HORA_SALIDA_CORREGIDA'] = pd.to_datetime(
            fecha_real.astype(str) + ' ' + hora_real.astype(str), errors='coerce'
        )
        print("✓ Corrección Temporal completada (Mitigación del desfase de 1899).")

    # =========================================================================
    # PASO 3 — NORMALIZACIÓN FINANCIERA
    # =========================================================================
    # Los campos 'valor_pactado' y 'valor_pagado' registran el flete acordado
    # y el flete efectivamente pagado al transportador. En el RNDC pueden
    # llegar con formato de texto (ej: "$ 1.250.000" o "1250000,00"), lo que
    # impide cualquier comparación numérica.
    #
    # RELEVANCIA CON EL PDF (Slide SICE-TAC):
    # El Decreto 1017 convirtió el SICE-TAC en un piso legal: pagar por debajo
    # del costo eficiente es una infracción sancionable con multas de hasta
    # $207.6 millones COP. Para detectar esos casos, los valores deben ser
    # comparables como números flotantes.
    #
    # El regex r'[^\d.]' elimina todo lo que no sea dígito o punto decimal,
    # luego pd.to_numeric convierte a float. errors='coerce' convierte
    # cualquier valor no convertible en NaN en lugar de lanzar una excepción.
    columnas_financieras = ['valor_pactado', 'valor_pagado']
    for col in columnas_financieras:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
    print("✓ Normalización Financiera completada.")

    # =========================================================================
    # PASO 4 — INGENIERÍA DE EQUIDAD: ATRIBUTO PROTEGIDO ESPACIAL (A)
    # =========================================================================
    # CONTEXTO DEL PDF (Slides de Nodos de Transferencia y Mega-Proyectos):
    # Colombia está invirtiendo en nodos intermodales estratégicos donde la
    # carga cambia de modo (carretera → tren → río) sin fricción: La Dorada
    # conecta la vía férrea con la hidrovía del Magdalena; Cartagena y
    # Barranquilla son los puertos de salida oceánica; Chiriguaná y Puerto
    # Salgar son puntos críticos de transferencia. Sin embargo, el 78-80% del
    # transporte sigue siendo carretero porque los algoritmos de asignación
    # de carga fueron entrenados con datos históricos que ignoran estas rutas.
    #
    # CONSTRUCCIÓN DEL ATRIBUTO:
    # El código DIVIPOLA (División Político-Administrativa de Colombia) identifica
    # municipios con un código numérico. Los primeros 2 dígitos son el
    # departamento y los siguientes 3 son el municipio.
    #
    # CORRECCIÓN RESPECTO AL CÓDIGO ORIGINAL:
    # El código original hacía: .str.zfill(8).str[:5]
    # Esto rellena hasta 8 ceros y luego toma 5 — INCORRECTO porque toma
    # los primeros 5 caracteres del string con padding, no el código real.
    # Ejemplo: el código "17380" primero se convierte en "00017380" y luego
    # se recorta a "00017" → ¡nunca coincidirá con '17380' en el diccionario!
    #
    # La forma correcta es: tomar los primeros 5 caracteres del código real
    # y LUEGO rellenar a 5 si viniera un código más corto de lo esperado.
    if 'codigo_descargue' in df.columns:
        df['DIVIPOLA_RAIZ'] = (
            df['codigo_descargue']
            .astype(str)
            .str.split('.')   # Elimina decimales si el código llegó como float (ej: "17380.0")
            .str[0]           # Toma la parte entera
            .str[:5]          # ✅ CORRECTO: primero recorta a 5 dígitos reales
            .str.zfill(5)     # ✅ CORRECTO: luego rellena a 5 si el código es más corto
        )

        # Nodos multimodales estratégicos 2026 (fuente: PDF Colombia Logistics Blueprint):
        # 17380 → La Dorada (Caldas): epicentro terrestre, conecta férreo con hidrovía
        # 20178 → Chiriguaná (Cesar): nodo férreo del proyecto APP La Dorada–Chiriguaná
        # 25572 → Puerto Salgar (Cundinamarca): punto de transferencia carretera–río
        # 13001 → Cartagena (Bolívar): puerto oceánico de salida, convergencia río + 4G/5G
        nodos_multimodales = ['17380', '20178', '25572', '13001']

        # A=0: Nodo Intermodal/Periferia → grupo PROTEGIDO (potencialmente discriminado)
        # A=1: Ruta Carretera Central → grupo FAVORECIDO (mayoritario en datos históricos)
        # np.where(condición, valor_si_verdadero, valor_si_falso)
        df['A_intermodal'] = np.where(
            df['DIVIPOLA_RAIZ'].isin(nodos_multimodales), 0, 1
        )
        print("✓ Atributo Protegido (A) configurado por segmentación geográfica.")

    # =========================================================================
    # PASO 5 — INGENIERÍA DE EQUIDAD: VARIABLE OBJETIVO OPERATIVA (Y)
    # =========================================================================
    # CONTEXTO DEL PDF (Slide SuperTransporte / Sanciones):
    # No pagar las horas de espera (standby) en cargue y descargue es una
    # infracción sancionable. Estas horas de espera son también el síntoma
    # más medible de la ineficiencia estructural: si un nodo intermodal recibe
    # sistemáticamente más horas de espera, es evidencia de sesgo territorial.
    #
    # DEFINICIÓN DEL UMBRAL (Tau):
    # Se usa el percentil 75 como umbral crítico. Esto significa que Y=1
    # identifica el 25% de viajes con mayor tiempo de espera en descargue,
    # que son los casos de ineficiencia más severa.
    #
    # CORRECCIÓN RESPECTO AL CÓDIGO ORIGINAL:
    # En el código original, el tau se calculaba ANTES del dropna del Paso 6.
    # Esto contamina el umbral: registros con datos incompletos (que serán
    # descartados) distorsionan la distribución del percentil 75.
    # La corrección es limpiar primero horas_espera_descargue de NaN antes
    # de calcular tau, para que refleje solo los datos válidos del estudio.
    if 'horas_espera_descargue' in df.columns:
        df['horas_espera_descargue'] = pd.to_numeric(
            df['horas_espera_descargue'], errors='coerce'
        )

        # ✅ CORRECTO: calcular tau solo sobre valores no nulos,
        # para que el percentil no se vea afectado por registros inválidos
        # que serán descartados en el Paso 6.
        tau = df['horas_espera_descargue'].dropna().quantile(0.75)

        # Y=1 si las horas de espera superan el umbral crítico (ineficiencia)
        # Y=0 si las horas de espera están dentro del rango estándar
        df['Y_riesgo'] = (df['horas_espera_descargue'] > tau).astype(int)
        print(f"✓ Variable Objetivo (Y) configurada. Umbral crítico (Tau 75%): {tau:.2f} horas.")

    # =========================================================================
    # PASO 6 — INTEGRIDAD DE GRUPOS: LIMPIEZA FINAL Y EDA INICIAL
    # =========================================================================
    # Se eliminan registros que tengan NaN en cualquiera de las tres variables
    # críticas para el estudio. Un registro sin A, sin Y, o sin horas_espera
    # no puede usarse ni para entrenar ni para evaluar el modelo de equidad.
    #
    # La list comprehension [col for col in ... if col in df.columns] es
    # una guardia defensiva: si una columna no pudo construirse en pasos
    # anteriores (por ejemplo, porque el CSV no tenía 'codigo_descargue'),
    # no lanza un KeyError sino que simplemente la omite del dropna.
    variables_criticas = ['A_intermodal', 'Y_riesgo', 'horas_espera_descargue']
    df_limpio = df.dropna(
        subset=[col for col in variables_criticas if col in df.columns]
    )

    # --- REPORTE EDA INICIAL ---
    print("\n=== REPORTE EDA INICIAL (GRUPO 1) ===")
    print(f"Total de registros viables tras limpieza: {df_limpio.shape[0]}")

    # Distribución del Atributo Protegido:
    # Esperamos ver una fuerte mayoría en A=1 (rutas carreteras centrales),
    # lo que reflejaría la matriz modal del PDF: 78-80% carretero.
    # Una proporción muy baja de A=0 confirmaría la hipótesis de que los
    # nodos intermodales están subrepresentados en los datos históricos.
    print("\nDistribución del Atributo Protegido (A):")
    dist_A = df_limpio['A_intermodal'].value_counts(normalize=True) * 100
    print(f"  Rutas Centrales Tradicionales (A=1): {dist_A.get(1, 0):.2f}%")
    print(f"  Nodos Intermodales/Periferia  (A=0): {dist_A.get(0, 0):.2f}%")

    # Distribución de la Variable Objetivo:
    # Por construcción (percentil 75), esperamos ~25% Y=1 y ~75% Y=0.
    # Una desviación significativa de ese 25/75 entre grupos A=0 y A=1
    # sería evidencia preliminar de sesgo territorial.
    print("\nDistribución de la Variable Objetivo (Y):")
    dist_Y = df_limpio['Y_riesgo'].value_counts(normalize=True) * 100
    print(f"  Operación Estándar     (Y=0): {dist_Y.get(0, 0):.2f}%")
    print(f"  Ineficiencia Crítica   (Y=1): {dist_Y.get(1, 0):.2f}%")

    # Exportación del dataset limpio para la Semana 2 (modelamiento).
    # Este CSV será el insumo para entrenar el clasificador de riesgo
    # y calcular métricas de equidad (Disparate Impact, Equal Opportunity).
    df_limpio.to_csv('RNDC_Grupo1_Limpio_S1.csv', index=False)
    print("\nArchivo 'RNDC_Grupo1_Limpio_S1.csv' exportado para la fase de modelamiento.")

    return df_limpio


# =========================================================================
# EJECUCIÓN
# =========================================================================
# Ajusta la ruta al archivo CSV de microdatos del RNDC en tu entorno local.
# Ejemplo Windows : df_final_g1 = preprocesar_rndc_grupo1('C:/datos/muestra_rndc.csv')
# Ejemplo Mac/Linux: df_final_g1 = preprocesar_rndc_grupo1('/home/usuario/muestra_rndc.csv')
# df_final_g1 = preprocesar_rndc_grupo1('muestra_rndc.csv')