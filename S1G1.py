import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def preprocesar_rndc_grupo1(ruta_archivo):
    """
    Ejecuta el pipeline de limpieza (Semana 1) para el análisis de sesgo territorial
    en el transporte de carga, aislando la ceguera algorítmica intermodal.
    """
    print("=== INICIANDO PROTOCOLO DE LIMPIEZA RNDC - GRUPO 1 ===")
    
    # 1. Carga del conjunto de datos de microdatos reales
    try:
        df = pd.read_csv(ruta_archivo, sep=',', quotechar='"', low_memory=False)
        print(f"Datos originales cargados: {df.shape[0]} registros.")
    except Exception as e:
        return f"Error al cargar el archivo: {e}"

    # 2. PROTOCOLO DE LIMPIEZA: Corrección Temporal (Desfase 1899)
    # Se fusiona la fecha correcta de 'fechasalidacargue' con la hora de 'hora_salida_cargue'
    if 'fechasalidacargue' in df.columns and 'hora_salida_cargue' in df.columns:
        fecha_real = pd.to_datetime(df['fechasalidacargue'], errors='coerce').dt.date
        hora_real = pd.to_datetime(df['hora_salida_cargue'], errors='coerce').dt.time
        df['FECHA_HORA_SALIDA_CORREGIDA'] = pd.to_datetime(fecha_real.astype(str) + ' ' + hora_real.astype(str), errors='coerce')
        print("✓ Corrección Temporal completada (Mitigación del desfase de 1899).")

    # 3. PROTOCOLO DE LIMPIEZA: Normalización Financiera
    # Aunque el Grupo 1 se enfoca en tiempo, la limpieza integral es obligatoria
    columnas_financieras = ['valor_pactado', 'valor_pagado']
    for col in columnas_financieras:
        if col in df.columns:
            # Se eliminan caracteres no numéricos y se convierte a flotante
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    print("✓ Normalización Financiera completada.")

    # 4. INGENIERÍA DE EQUIDAD: Atributo Protegido Espacial (A)
    # Extracción de la raíz DIVIPOLA de 5 dígitos desde 'codigo_descargue'
    if 'codigo_descargue' in df.columns:
        df['DIVIPOLA_RAIZ'] = df['codigo_descargue'].astype(str).str.split('.').str[0].str.zfill(8).str[:5]
        
        # Nodos multimodales 2026: La Dorada (17380), Chiriguaná (20178), Puerto Salgar (25572), Cartagena (13001)
        nodos_multimodales = ['17380', '20178', '25572', '13001']
        
        # A=0: Periferia/Intermodal (Protegido), A=1: Central/Carretero (Favorecido)
        df['A_intermodal'] = np.where(df['DIVIPOLA_RAIZ'].isin(nodos_multimodales), 0, 1)
        print("✓ Atributo Protegido (A) configurado por segmentación geográfica.")

    # 5. INGENIERÍA DE EQUIDAD: Variable Objetivo Operativa (Y)
    # Formalización de la ineficiencia estructural que causa el sesgo
    if 'horas_espera_descargue' in df.columns:
        df['horas_espera_descargue'] = pd.to_numeric(df['horas_espera_descargue'], errors='coerce')
        
        # Cálculo del percentil crítico 75 (Tau)
        tau = df['horas_espera_descargue'].quantile(0.75)
        
        # Y=1 si horas_espera > Tau, de lo contrario Y=0
        df['Y_riesgo'] = (df['horas_espera_descargue'] > tau).astype(int)
        print(f"✓ Variable Objetivo (Y) configurada. Umbral crítico (Tau 75%): {tau:.2f} horas.")

    # 6. PROTOCOLO DE LIMPIEZA: Integridad de Grupos
    # Eliminación estratificada de registros nulos en las variables críticas del estudio
    variables_criticas = ['A_intermodal', 'Y_riesgo', 'horas_espera_descargue']
    df_limpio = df.dropna(subset=[col for col in variables_criticas if col in df.columns])
    
    print("\n=== REPORTE EDA INICIAL (GRUPO 1) ===")
    print(f"Total de registros viables tras limpieza: {df_limpio.shape[0]}")
    
    print("\nDistribución del Atributo Protegido (A):")
    dist_A = df_limpio['A_intermodal'].value_counts(normalize=True) * 100
    print(f"Rutas Centrales Tradicionales (A=1): {dist_A.get(1, 0):.2f}%")
    print(f"Nodos Intermodales/Periferia (A=0): {dist_A.get(0, 0):.2f}%")
    
    print("\nDistribución de la Variable Objetivo (Y):")
    dist_Y = df_limpio['Y_riesgo'].value_counts(normalize=True) * 100
    print(f"Operación Estándar (Y=0): {dist_Y.get(0, 0):.2f}%")
    print(f"Ineficiencia Crítica (Y=1): {dist_Y.get(1, 0):.2f}%")
    
    # Exportación del Dataset Limpio para la Semana 2
    df_limpio.to_csv('RNDC_Grupo1_Limpio_S1.csv', index=False)
    print("\nArchivo 'RNDC_Grupo1_Limpio_S1.csv' exportado exitosamente para la fase de modelamiento.")
    
    return df_limpio

# Ejecución de la función (Los estudiantes deben ajustar la ruta a su archivo local)
# df_final_g1 = preprocesar_rndc_grupo1('muestra_rndc.csv')