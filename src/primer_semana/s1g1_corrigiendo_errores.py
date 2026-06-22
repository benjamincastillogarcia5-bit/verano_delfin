import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calcular_haversine_vectorizado(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def preprocesar_rndc_grupo1(ruta_archivo):
    print("=== INICIANDO PROTOCOLO DE LIMPIEZA RNDC - GRUPO 1 ===")

    # 1. Carga del conjunto de datos
    try:
        df = pd.read_csv(ruta_archivo, sep=',', quotechar='"', low_memory=False)
        print(f"Datos originales cargados: {df.shape[0]} registros y {df.shape[1]} columnas/características. \n")
        df.columns = df.columns.str.lower()
        print(f"\t Columnas del DF \n \n {df.columns.tolist()} \n")

        # === CAMBIO 1: ELIMINACIÓN DE NULOS AHORA ES TRANSPARENTE ===
        columnas_a_limpiar = [col for col in ['unid_medida', 'descargue'] if col in df.columns]
        if columnas_a_limpiar:
            print("--- Reporte de nulos en columnas críticas (antes de filtrar) ---")
            for col in columnas_a_limpiar:
                n_nulos = df[col].isna().sum()
                pct = (n_nulos / len(df)) * 100 if len(df) > 0 else 0
                print(f"  '{col}': {n_nulos} nulos ({pct:.2f}% del total)")

            filas_antes = df.shape[0]
            df = df.dropna(subset=columnas_a_limpiar)
            filas_eliminadas = filas_antes - df.shape[0]
            pct_eliminado = (filas_eliminadas / filas_antes) * 100 if filas_antes > 0 else 0
            print(f"✓ Se eliminaron {filas_eliminadas} registros ({pct_eliminado:.2f}%) por nulos en {columnas_a_limpiar}.")
            print(f"  Quedan {df.shape[0]} registros tras este filtro.\n")

    except Exception as e:
        return f"Error al cargar el archivo: {e}"

    # 2. Corrección Temporal
    if 'fechasalidacargue' in df.columns and 'hora_salida_cargue' in df.columns:
        fecha_real = pd.to_datetime(df['fechasalidacargue'], errors='coerce').dt.date
        hora_real = pd.to_datetime(df['hora_salida_cargue'], errors='coerce').dt.time
        df['FECHA_HORA_SALIDA_CORREGIDA'] = pd.to_datetime(fecha_real.astype(str) + ' ' + hora_real.astype(str), errors='coerce')
        print("✓ Corrección Temporal completada (Mitigación del desfase de 1899).")

    # 3. Normalización Financiera (Incluyendo Min-Max silencioso)
    if 'valor_pactado' in df.columns:
        df['valor_pactado'] = pd.to_numeric(df['valor_pactado'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        v_min = df['valor_pactado'].min()
        v_max = df['valor_pactado'].max()
        df['valor_pactado_norm'] = (df['valor_pactado'] - v_min) / (v_max - v_min) if v_max > v_min else 0.0
    print("✓ Normalización Financiera completada.\n")

    # 3.5. Normalización de Masas a Kilogramos (CRUCE CON DENSIDAD DE PRODUCTOS)
    print("Normalizando unidades de medida de carga a Kilogramos (basado en densidad del producto)...")
    if 'cantidad' in df.columns and 'unid_medida' in df.columns and 'producto' in df.columns:
        df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce').fillna(0)
        df['unid_medida'] = df['unid_medida'].astype(str).str.upper().str.strip()
        mascara_galones = df['unid_medida'] == 'GALONES'

        if mascara_galones.sum() > 0:
            # 1. Definir función de mapeo de densidades (kg por galón) calibrada con NLP
            def estimar_densidad(nombre_producto):
                nombre = str(nombre_producto).lower()

                # 1. Combustibles y Petróleo (Tus keywords: combustibles, combustible, petróleo)
                if any(x in nombre for x in ['combustible', 'petróleo', 'petroleo', 'gasolina', 'acpm']):
                    return 3.15

                # 2. Aceites y Grasas (Tus keywords: aceites, aceite, grasas)
                elif any(x in nombre for x in ['aceite', 'grasa']):
                    return 3.40

                # 3. Lácteos (Tus keywords: leche, lácteos)
                elif any(x in nombre for x in ['leche', 'lácteo', 'lacteo']):
                    return 3.90

                # 4. Químicos y Abonos (Tus keywords: químicas, químicos, quimicos, abonos)
                # (Asumimos una densidad ligeramente mayor al agua para soluciones químicas industriales)
                elif any(x in nombre for x in ['química', 'quimica', 'químico', 'quimico', 'abono']):
                    return 4.10

                # 5. Bebidas, Agua y Alcoholes (Tus keywords: agua, bebidas, vinagre, alcohólicos, cerveza, gasificada)
                elif any(x in nombre for x in ['agua', 'bebida', 'vinagre', 'alcohólico', 'cerveza', 'gasificada']):
                    return 3.785

                # Por defecto: Líquidos generales
                else:
                    return 3.785

            # 2. Aplicar la función solo a los registros que están en galones
            df.loc[mascara_galones, 'densidad_estimada'] = df.loc[mascara_galones, 'producto'].apply(estimar_densidad)

            # 3. Realizar la conversión matemática: Galones * Densidad = Kilogramos
            df.loc[mascara_galones, 'cantidad'] = df.loc[mascara_galones, 'cantidad'] * df.loc[mascara_galones, 'densidad_estimada']
            df.loc[mascara_galones, 'unid_medida'] = 'KILOGRAMOS'

            # 4. Generar el CSV de auditoría para el equipo de investigación
            # Aisla los productos únicos que estaban en galones y su densidad asignada
            df_auditoria = df.loc[mascara_galones, ['producto', 'densidad_estimada']].drop_duplicates().sort_values('producto')
            df_auditoria.to_csv('Auditoria_Densidades_Galones.csv', index=False)

            # 5. Limpiar la columna temporal para no ensuciar el dataset final de Machine Learning
            df = df.drop(columns=['densidad_estimada'])

            print(f"✓ Se convirtieron {mascara_galones.sum()} registros de Galones a Kg usando densidades específicas.")
            print(f"✓ Se generó el archivo 'Auditoria_Densidades_Galones.csv' con {len(df_auditoria)} productos únicos para revisión.\n")
        else:
            print("✓ No se encontraron registros en Galones para convertir.\n")

        df['unid_medida'] = df['unid_medida'].replace('NAN', 'KILOGRAMOS')
    # 3.6. CÁLCULO DE DISTANCIA ESTIMADA (HAVERSINE)
    if 'cargue' in df.columns and 'descargue' in df.columns:
        # Diccionario de coordenadas actualizado con formato "CIUDAD DEPARTAMENTO" del RNDC
        coordenadas = {
            # Nodos de tu estudio (Formato RNDC)
            'CARTAGENA BOLIVAR': (10.3997, -75.5144),
            'LA DORADA CALDAS': (5.4528, -74.6719),
            'CHIRIGUANA CESAR': (9.3608, -73.5999),
            'CHIRIGUANÁ CESAR': (9.3608, -73.5999),
            'PUERTO SALGAR CUNDINAMARCA': (5.4630, -74.6548),
            'RINCON HONDO CHIRIGUANA CESAR': (9.3608, -73.5999),
            'BAYUNCA CARTAGENA BOLIVAR': (10.5500, -75.4333),
            
            # Top Capitales y Nodos Logísticos (Formato RNDC)
            'BOGOTA BOGOTA D. C.': (4.6097, -74.0817),
            'MEDELLIN ANTIOQUIA': (6.2442, -75.5812),
            'BARRANQUILLA ATLANTICO': (10.9685, -74.7813),
            'BUENAVENTURA VALLE DEL CAUCA': (3.8801, -77.0311),
            'CALI VALLE DEL CAUCA': (3.4516, -76.5319),
            'YUMBO VALLE DEL CAUCA': (3.5833, -76.4953),
            'FUNZA CUNDINAMARCA': (4.7167, -74.2114),
            'BUCARAMANGA SANTANDER': (7.1193, -73.1227),
            'PEREIRA RISARALDA': (4.8087, -75.6906),
            'TOCANCIPA CUNDINAMARCA': (4.9650, -73.9140),
            'SANTA MARTA MAGDALENA': (11.2408, -74.1990),
            'MANIZALES CALDAS': (5.0689, -75.5174),
            'CUCUTA NORTE DE SANTANDER': (7.8939, -72.5078),
            'COTA CUNDINAMARCA': (4.8105, -74.0991),
            'PALMIRA VALLE DEL CAUCA': (3.5394, -76.3036),
            'IBAGUE TOLIMA': (4.4389, -75.2322),
            'ITAGUI ANTIOQUIA': (6.1732, -75.6030),
            'RIONEGRO ANTIOQUIA': (6.1551, -75.3737),
            'GIRON SANTANDER': (7.0682, -73.1698),
            'MOSQUERA CUNDINAMARCA': (4.7059, -74.2302),
            'DOSQUEBRADAS RISARALDA': (4.8398, -75.6722),
            'ENVIGADO ANTIOQUIA': (6.1759, -75.5917),
            'GUADALAJARA DE BUGA VALLE DEL CAUCA': (3.9000, -76.3000),
            'BARRANCABERMEJA SANTANDER': (7.0653, -73.8547),
            'NEIVA HUILA': (2.9273, -75.2819),
            'VILLAVICENCIO META': (4.1420, -73.6266),
            'MALAMBO ATLANTICO': (10.8583, -74.7774),

            # --- TOP 20 CIUDADES FALTANTES (Nuevas Coordenadas) ---
            'MADRID CUNDINAMARCA': (4.7325, -74.2644),
            'TENJO CUNDINAMARCA': (4.8719, -74.1453),
            'GIRARDOTA ANTIOQUIA': (6.3755, -75.4461),
            'MONTERIA CORDOBA': (8.7514, -75.8814),
            'PASTO NARINO': (1.2136, -77.2811),
            'LA ESTRELLA LA ESTRELLA ANTIOQUIA': (6.1578, -75.6431),
            'ARMENIA QUINDIO': (4.5339, -75.6811),
            'GALAPA ATLANTICO': (10.8994, -74.8814),
            'VALLEDUPAR CESAR': (10.4631, -73.2531),
            'YOPAL CASANARE': (5.3378, -72.3958),
            'SOGAMOSO BOYACA': (5.7144, -72.9339),
            'SOACHA CUNDINAMARCA': (4.5781, -74.2144),
            'FACATATIVA CUNDINAMARCA': (4.8136, -74.3536),
            'BELLO ANTIOQUIA': (6.3331, -75.5583),
            'CAJICA CUNDINAMARCA': (4.9181, -74.0289),
            'CARTAGO VALLE DEL CAUCA': (4.6989, -75.9142),
            'SABANETA ANTIOQUIA': (6.1517, -75.6153),
            'SONSON ANTIOQUIA': (5.7114, -75.3111),
            'TUNJA BOYACA': (5.5353, -73.3678),
            'POPAYAN CAUCA': (2.4411, -76.6061),
            
            # Versiones cortas por si alguna viene limpia
            'CARTAGENA': (10.3997, -75.5144),
            'BOGOTA': (4.6097, -74.0817)
        }
        cargue_limpio = df['cargue'].astype(str).str.upper().str.strip()
        descargue_limpio = df['descargue'].astype(str).str.upper().str.strip()

        # === CAMBIO 3: REPORTE DE COBERTURA DE COORDENADAS ===
        ciudades_usadas = pd.concat([cargue_limpio, descargue_limpio])
        en_diccionario = ciudades_usadas.isin(coordenadas.keys())

        conteo_con_coords = ciudades_usadas[en_diccionario].value_counts()
        conteo_sin_coords = ciudades_usadas[~en_diccionario].value_counts()

        print("--- Cobertura del diccionario de coordenadas ---")
        print(f"Ciudades CON coordenadas registradas ({conteo_con_coords.shape[0]} nombres distintos, "
              f"{en_diccionario.sum()} apariciones en cargue+descargue):")
        print(conteo_con_coords.to_string())

        print(f"\nCiudades SIN coordenadas registradas ({conteo_sin_coords.shape[0]} nombres distintos, "
              f"{(~en_diccionario).sum()} apariciones en cargue+descargue):")
        if conteo_sin_coords.empty:
            print("  (ninguna — diccionario completo para este dataset)")
        else:
            print(conteo_sin_coords.to_string())
            print("\n  → Agrega estos nombres (con sus coordenadas) al diccionario 'coordenadas' "
                  "y vuelve a correr el script para reducir el uso de la mediana de relleno.")
        print()

        df['lat_origen'] = cargue_limpio.map(lambda x: coordenadas.get(x, (np.nan, np.nan))[0])
        df['lon_origen'] = cargue_limpio.map(lambda x: coordenadas.get(x, (np.nan, np.nan))[1])
        df['lat_destino'] = descargue_limpio.map(lambda x: coordenadas.get(x, (np.nan, np.nan))[0])
        df['lon_destino'] = descargue_limpio.map(lambda x: coordenadas.get(x, (np.nan, np.nan))[1])

        df['distancia_estimada_km'] = calcular_haversine_vectorizado(
            df['lat_origen'], df['lon_origen'], df['lat_destino'], df['lon_destino']
        )
        n_rellenados = df['distancia_estimada_km'].isna().sum()
        df['distancia_estimada_km'] = df['distancia_estimada_km'].fillna(df['distancia_estimada_km'].median())
        print(f"✓ {n_rellenados} registros sin coordenadas fueron rellenados con la mediana de distancia.\n")

    # 4. Atributo Protegido (A)
    if 'codigo_descargue' in df.columns:
        df['divipola_raiz'] = df['codigo_descargue'].astype(str).str.split('.').str[0].str.zfill(8).str[:5]
        print("\t Zonas de descargue a considerar en base a su código de descargue \n")
        print(f"La Dorada (17380) = {df['descargue'].loc[df['divipola_raiz'] == '17380'].unique()}")
        print(f"Chiriguaná (20178) = {df['descargue'].loc[df['divipola_raiz'] == '20178'].unique()}")
        print(f"Puerto Salgar (25572) = {df['descargue'].loc[df['divipola_raiz'] == '25572'].unique()}")
        print(f"Cartagena (13001) = {df['descargue'].loc[df['divipola_raiz'] =='13001'].unique()}\n")

        nodos_multimodales = ['17380', '20178', '25572', '13001']
        df['a_intermodal'] = np.where(df["codigo_descargue"].isna(), np.nan,
                                      np.where(df['divipola_raiz'].isin(nodos_multimodales), 0, 1))
        print("✓ Atributo Protegido (A) configurado por segmentación geográfica.")

    # 5. Variable Objetivo (Y)
    if 'horas_espera_descargue' in df.columns:
        df['horas_espera_descargue'] = pd.to_numeric(df['horas_espera_descargue'], errors='coerce')
        tau = df['horas_espera_descargue'].quantile(0.75)
        df['y_riesgo'] = np.where(df['horas_espera_descargue'].isna(), np.nan,
                                  (df['horas_espera_descargue'] > tau).astype(float))
        print(f"✓ Variable Objetivo (Y) configurada. Umbral crítico (Tau 75%): {tau:.2f} horas.\n")

    # 6. Integridad de Grupos e Impresión de Reporte
    variables_criticas = ['a_intermodal', 'y_riesgo', 'horas_espera_descargue']
    print("Se analizarán las VARIABLES CRÍTICAS (A, Y y horas de espera en descargue)\n")
    print("Valores nulos en cada variable crítica:")
    print(f"Horas espera descargue: {df['horas_espera_descargue'].isna().sum()}")
    print(f"A intermodal: {df['a_intermodal'].isna().sum()}")
    print(f"Y riesgo: {df['y_riesgo'].isna().sum()}\n")

    print("Se puede ver que las variables: 'codigo_descargue' y 'horas_esperas_descargue' son OBLIGATORIAS en la base de datos, y no hay ningún valor NULO.\n")

    print("Análisis de las Horas de espera:")
    print(f"% de registros con espera = 0: \t {(df['horas_espera_descargue'] == 0).mean() * 100:.2f}%\n")

    print("Horas de espera = 0 separada por A intermodal \n A=0: Periferia/Intermodal (Protegido) \n A=1: Central/Carretero (Favorecido)")
    serie_ceros = df.groupby('a_intermodal')['horas_espera_descargue'].apply(lambda x: (x == 0).mean() * 100)
    print("a_intermodal")
    print(f"0.0    {serie_ceros.get(0, 0):.6f}")
    print(f"1.0    {serie_ceros.get(1, 0):.6f}")
    print("Name: horas_espera_descargue, dtype: float64\n")

    diferencia_porcentual = serie_ceros.get(0, 0) - serie_ceros.get(1, 0)
    print(f"Hay aproximadamente {diferencia_porcentual:.1f}% más reportes de 0 en el A=0 que en A=1")
    print("(No se sabe si esto es por la zona, o por la empresa)\n")

    df_limpio = df.dropna(subset=[col for col in variables_criticas if col in df.columns])

    print("=== REPORTE EDA INICIAL (GRUPO 1) ===")
    print(f"Total de registros viables tras limpieza: {df_limpio.shape[0]}\n")

    print("Distribución del Atributo Protegido (A):")
    dist_A = df_limpio['a_intermodal'].value_counts(normalize=True) * 100
    print(f"Rutas Centrales Tradicionales (A=1): {dist_A.get(1, 0):.2f}%")
    print(f"Nodos Intermodales/Periferia (A=0): {dist_A.get(0, 0):.2f}%\n")

    print("Distribución de la Variable Objetivo (Y):")
    dist_Y = df_limpio['y_riesgo'].value_counts(normalize=True) * 100
    print(f"Operación Estándar (Y=0): {dist_Y.get(0, 0):.2f}%")
    print(f"Ineficiencia Crítica (Y=1): {dist_Y.get(1, 0):.2f}%\n")

    # Exportación
    df_limpio.to_csv('RNDC_Grupo1_Limpio_S1.csv', index=False)
    print("Archivo 'RNDC_Grupo1_Limpio_S1.csv' exportado exitosamente para la fase de modelamiento.")

    return df_limpio

# Ejecución
df_final_g1 = preprocesar_rndc_grupo1("Tiempos_Logísticos_de_cada_viaje_de_vehículos_de_carga_20260615.csv")