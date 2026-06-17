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
        print(f"Datos originales cargados: {df.shape[0]} registros y {df.shape[1]} columnas/características.")
        df.columns = df.columns.str.lower()
        print(df.columns)
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
        else:
          print("no se encontraron columnas financieras en DF")
    print("✓ Normalización Financiera completada.")

preprocesar_rndc_grupo1("Tiempos_Logísticos_de_cada_viaje_de_vehículos_de_carga_20260615.csv")

