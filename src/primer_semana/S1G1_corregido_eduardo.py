import pandas as pd
import numpy as np
import warnings
from pathlib import Path

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
        print(f"Datos originales cargados: {df.shape[0]} registros y {df.shape[1]} columnas/características. \n")
        df.columns = df.columns.str.lower()
        print(f"\t Columnas del DF \n \n {df.columns} \n")
    except Exception as e:
        print(f"\n[!] ERROR CRÍTICO AL CARGAR EL ARCHIVO:\n{e}")
        print(f"Ruta intentada: {ruta_archivo}")
        return None

    # 2. PROTOCOLO DE LIMPIEZA: Corrección Temporal (Desfase 1899)
    if 'fechasalidacargue' in df.columns and 'hora_salida_cargue' in df.columns:
        fecha_real = pd.to_datetime(df['fechasalidacargue'], errors='coerce').dt.date
        hora_real = pd.to_datetime(df['hora_salida_cargue'], errors='coerce').dt.time
        df['FECHA_HORA_SALIDA_CORREGIDA'] = pd.to_datetime(fecha_real.astype(str) + ' ' + hora_real.astype(str), errors='coerce')
        print("✓ Corrección Temporal completada (Mitigación del desfase de 1899).")

    # 3. PROTOCOLO DE LIMPIEZA: Normalización Financiera
    columnas_financieras = ['valor_pactado', 'valor_pagado']
    for col in columnas_financieras:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        else:
            print("No se encontraron columnas financieras en DF")
    print("\n✓ Normalización Financiera completada.\n")

    # 4. INGENIERÍA DE EQUIDAD: Atributo Protegido Espacial (A)
    if 'codigo_descargue' in df.columns:
        # Se inicializa como string para evitar pérdida de ceros a la izquierda y errores de tipo
        df['divipola_raiz'] = pd.Series(dtype='string')
        df['divipola_raiz'] = df['codigo_descargue'].astype(str).str.split('.').str[0].str.zfill(8).str[:5]
        
        print("\tZonas de descargue a considerar en base a su código de descargue\n")
        print(f"La Dorada (17380) = {df['descargue'].loc[df['divipola_raiz'] == '17380'].unique()}")
        print(f"Chiriguaná (20178) = {df['descargue'].loc[df['divipola_raiz'] == '20178'].unique()}")
        print(f"Puerto Salgar (25572) = {df['descargue'].loc[df['divipola_raiz'] == '25572'].unique()}")
        print(f"Cartagena (13001) = {df['descargue'].loc[df['divipola_raiz'] == '13001'].unique()}\n")

        # Nodos multimodales 2026
        nodos_multimodales = ['17380', '20178', '25572', '13001']
        
        # A=0: Periferia/Intermodal (Protegido), A=1: Central/Carretero (Favorecido)
        df['a_intermodal'] = np.where(df['divipola_raiz'].isin(nodos_multimodales), 0, 1)
        print("✓ Atributo Protegido (A) configurado por segmentación geográfica.")

    # 5. INGENIERÍA DE EQUIDAD: Variable Objetivo Operativa (Y)
    if 'horas_espera_descargue' in df.columns:
        df['horas_espera_descargue'] = pd.to_numeric(df['horas_espera_descargue'], errors='coerce')
        tau = df['horas_espera_descargue'].quantile(0.75)
        df['y_riesgo'] = (df['horas_espera_descargue'] > tau).astype(int)
        print(f"✓ Variable Objetivo (Y) configurada. Umbral crítico (Tau 75%): {tau:.2f} horas.")

    # 6. PROTOCOLO DE LIMPIEZA: Integridad de Grupos
    variables_criticas = ['a_intermodal', 'y_riesgo', 'horas_espera_descargue']
    df_limpio = df.dropna(subset=[col for col in variables_criticas if col in df.columns])
    
    print("\n=== REPORTE EDA INICIAL (GRUPO 1) ===")
    print(f"Total de registros viables tras limpieza: {df_limpio.shape[0]}")
    
    if not df_limpio.empty:
        print("\nDistribución del Atributo Protegido (A):")
        dist_A = df_limpio['a_intermodal'].value_counts(normalize=True) * 100
        print(f"Rutas Centrales Tradicionales (A=1): {dist_A.get(1, 0):.2f}%")
        print(f"Nodos Intermodales/Periferia (A=0): {dist_A.get(0, 0):.2f}%")
        
        print("\nDistribución de la Variable Objetivo (Y):")
        dist_Y = df_limpio['y_riesgo'].value_counts(normalize=True) * 100
        print(f"Operación Estándar (Y=0): {dist_Y.get(0, 0):.2f}%")
        print(f"Ineficiencia Crítica (Y=1): {dist_Y.get(1, 0):.2f}%")

        # Exportación del Dataset Limpio en la misma carpeta del archivo original
        ruta_salida = Path(ruta_archivo).parent / 'RNDC_Grupo1_Limpio_S1.csv'
        df_limpio.to_csv(ruta_salida, index=False)
        print(f"\nArchivo exportado exitosamente en:\n{ruta_salida}")
    else:
        print("\nAdvertencia: El DataFrame quedó vacío tras la limpieza de variables críticas.")
    
    return df_limpio

# === CONFIGURACIÓN DE RUTAS DINÁMICAS ===
# 1. Detectamos la ubicación del script actual
ruta_actual = Path(__file__).resolve()

# 2. Retrocedemos hasta la carpeta raíz del proyecto (code_delfin)
ruta_raiz = ruta_actual.parent.parent.parent

# 3. Construimos la ruta exacta hacia la carpeta 'data' y el archivo CSV
ruta_data = ruta_raiz / 'data'
archivo_csv = ruta_data / 'RNDC_Grupo1_Limpio_S1.csv'
# 4. Ejecución
if __name__ == "__main__":
    df_final_g1 = preprocesar_rndc_grupo1(archivo_csv)