import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def preprocesar_rndc_grupo1(ruta_archivo):
    """
    Ejecuta el pipeline de limpieza (Semana 1) para el análisis de sesgo territorial
    en el transporte de carga.
    """
    print("=== INICIANDO PROTOCOLO DE LIMPIEZA RNDC - GRUPO 1 ===")
    
    try:
        # Carga del CSV. Se ajusta low_memory=False para microdatos masivos
        df = pd.read_csv(ruta_archivo, sep=',', quotechar='"', low_memory=False)
        print(f"Datos originales cargados: {df.shape[0]} registros.")
    except Exception as e:
        print(f"Error crítico al cargar el archivo: {e}")
        return None 

    # 1. Corrección Temporal (Desfase 1899)
    # Corrección de nomenclatura: FECHASALIDACARGUE y HORA_SALIDA_CARGUE
    if 'FECHASALIDACARGUE' in df.columns and 'HORA_SALIDA_CARGUE' in df.columns:
        mascara_fechas = df['FECHASALIDACARGUE'].notna() & df['HORA_SALIDA_CARGUE'].notna()
        df['FECHA_HORA_SALIDA_CORREGIDA'] = pd.NaT
        
        fechas_str = df.loc[mascara_fechas, 'FECHASALIDACARGUE'].astype(str)
        horas_str = df.loc[mascara_fechas, 'HORA_SALIDA_CARGUE'].astype(str)
        
        df.loc[mascara_fechas, 'FECHA_HORA_SALIDA_CORREGIDA'] = pd.to_datetime(
            fechas_str + ' ' + horas_str, errors='coerce'
        )
        print("✓ Corrección Temporal completada.")

    # 2. Normalización Financiera 
    # Nomenclatura confirmada: VALOR_PACTADO y VALOR_PAGADO
    columnas_financieras = ['VALOR_PACTADO', 'VALOR_PAGADO']
    for col in columnas_financieras:
        if col in df.columns:
            mascara_nulos = df[col].notna()
            df.loc[mascara_nulos, col] = pd.to_numeric(
                df.loc[mascara_nulos, col].astype(str).str.replace(r'[^\d.]', '', regex=True), 
                errors='coerce'
            )
    print("✓ Normalización Financiera completada.")

    # 3. Atributo Protegido Espacial (A)
    # Corrección de nomenclatura: CODIGO_DESCARGUE
    if 'CODIGO_DESCARGUE' in df.columns:
        # df['DIVIPOLA_RAIZ'] = np.nan
        df['DIVIPOLA_RAIZ'] = pd.Series(dtype='string')
        mascara_div = df['CODIGO_DESCARGUE'].notna()
        
        df.loc[mascara_div, 'DIVIPOLA_RAIZ'] = df.loc[mascara_div, 'CODIGO_DESCARGUE'] \
            .astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8).str[:5]
        
        # Nodos multimodales 2026: La Dorada (17380), Chiriguaná (20178), Puerto Salgar (25572), Cartagena (13001)
        nodos_multimodales = ['17380', '20178', '25572', '13001']
        
        # A=1: Red intermodal emergente, A=0: Carreteras tradicionales
        df['A_intermodal'] = np.where(
            df['DIVIPOLA_RAIZ'].isna(), 
            np.nan,  
            np.where(df['DIVIPOLA_RAIZ'].isin(nodos_multimodales), 1, 0)
        )
        print("✓ Atributo Protegido (A) configurado y blindado contra nulos.")

    # 4. Variable Objetivo Operativa (Y) 
    # Corrección de nomenclatura: HORAS_ESPERA_DESCARGUE
    if 'HORAS_ESPERA_DESCARGUE' in df.columns:
        df['HORAS_ESPERA_DESCARGUE'] = pd.to_numeric(df['HORAS_ESPERA_DESCARGUE'], errors='coerce')
        tau = df['HORAS_ESPERA_DESCARGUE'].quantile(0.75)
        
        df['Y_riesgo'] = np.where(
            df['HORAS_ESPERA_DESCARGUE'].isna(),
            np.nan,
            (df['HORAS_ESPERA_DESCARGUE'] > tau).astype(int)
        )
        print(f"✓ Variable Objetivo (Y) configurada. Umbral crítico (Tau 75%): {tau:.2f} horas.")

    # 5. Integridad de Grupos
    variables_criticas = ['A_intermodal', 'Y_riesgo', 'HORAS_ESPERA_DESCARGUE']
    df_limpio = df.dropna(subset=[col for col in variables_criticas if col in df.columns])
    
    print("\n=== REPORTE EDA INICIAL (GRUPO 1) ===")
    print(f"Total de registros viables tras limpieza: {df_limpio.shape[0]}")
    
    if not df_limpio.empty:
        print("\nDistribución del Atributo Protegido (A):")
        dist_A = df_limpio['A_intermodal'].value_counts(normalize=True) * 100
        print(f"Red Intermodal Emergente (A=1): {dist_A.get(1, 0):.2f}%")
        print(f"Carreteras Tradicionales (A=0): {dist_A.get(0, 0):.2f}%")
        
        print("\nDistribución de la Variable Objetivo (Y):")
        dist_Y = df_limpio['Y_riesgo'].value_counts(normalize=True) * 100
        print(f"Operación Estándar (Y=0): {dist_Y.get(0, 0):.2f}%")
        print(f"Ineficiencia Crítica (Y=1): {dist_Y.get(1, 0):.2f}%")
        
        df_limpio.to_csv('RNDC_Grupo1_Limpio_S1.csv', index=False)
        print("\nArchivo exportado exitosamente para la fase de modelamiento.")
    else:
        print("Advertencia: El DataFrame quedó vacío tras la limpieza de variables críticas.")
    
    return df_limpio                                                                          
# Define la ruta de tu archivo de datos original
ruta_del_archivo_original = 'C:/Users/bensa/Documents/code_delfin/data/RNDC_Grupo1_Limpio_S1.csv'
# Llama a la función y guarda el DataFrame limpio en una variable
df_limpio_final = preprocesar_rndc_grupo1(ruta_del_archivo_original)

# Si quieres ver las primeras filas del DataFrame limpio
if df_limpio_final is not None:
    print("\n--- Primeras 5 filas del DataFrame limpio ---")
    display(df_limpio_final.head())

# # Para descargar el archivo CSV que la función guardó en Colab
# from google.colab import files

# try:
#     files.download('RNDC_Grupo1_Limpio_S1.csv')
#     print("\nArchivo 'RNDC_Grupo1_Limpio_S1.csv' descargado exitosamente.")
# except Exception as e:
#     print(f"\nNo se pudo descargar el archivo: {e}. Asegúrate de que fue creado correctamente.")