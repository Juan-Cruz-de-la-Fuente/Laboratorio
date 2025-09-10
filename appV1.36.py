import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import io
from datetime import datetime
import base64
from scipy.spatial import Delaunay
from typing import Dict
import os
import zipfile
import random

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio de Aerodinámica y Fluidos - UTN HAEDO",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['figure.dpi'] = 200       # resolución en pantalla
plt.rcParams['savefig.dpi'] = 300      # resolución al exportar
plt.rcParams['figure.figsize'] = (12, 8)  # tamaño por defecto

# CSS personalizado para estilo moderno con nuevo color
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-teal: #08596C;
        --secondary-teal: #0A6B82;
        --accent-green: #10b981;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-800: #1f2937;
        --gray-900: #111827;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .header-container {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-teal) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(8, 89, 108, 0.3);
    }
    
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid var(--gray-200);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: var(--secondary-teal);
    }
    
    .metric-container {
        background: linear-gradient(135deg, var(--gray-100) 0%, white 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--gray-200);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-teal) 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(8, 89, 108, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(8, 89, 108, 0.4);
    }
    
    .selection-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .selection-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .selection-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .selection-card.selected {
        border-color: var(--primary-teal);
        background: linear-gradient(135deg, rgba(8, 89, 108, 0.1) 0%, rgba(10, 107, 130, 0.05) 100%);
    }
    
    .color-indicator {
        width: 100%;
        height: 8px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    
    .filter-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .step-indicator {
        background: var(--primary-teal);
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de la sesión
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = 'inicio'
if 'archivos_cargados' not in st.session_state:
    st.session_state.archivos_cargados = []
if 'datos_procesados' not in st.session_state:
    st.session_state.datos_procesados = {}
if 'configuracion_inicial' not in st.session_state:
    st.session_state.configuracion_inicial = {}
if 'sub_archivos_generados' not in st.session_state:
    st.session_state.sub_archivos_generados = {}
if 'datos_3d_filtrados' not in st.session_state:
    st.session_state.datos_3d_filtrados = {}
if 'sub_archivos_3d' not in st.session_state:
    st.session_state.sub_archivos_3d = {}
if 'configuracion_3d' not in st.session_state:
    st.session_state.configuracion_3d = {}
if 'sub_archivos_3d_generados' not in st.session_state:
    st.session_state.sub_archivos_3d_generados = {}
if 'diferencias_guardadas' not in st.session_state:
    st.session_state.diferencias_guardadas = {}

def extraer_tiempo_y_coordenadas(nombre_archivo):
    """Extraer tiempo y coordenadas X, Y del nombre del archivo.
    Soporta formatos nuevos como:
        XY_SapySync_X_1_Y_180_250811164701_10.CSV  -> tiempo=10, X=1, Y=180
    y formatos antiguos (T10s, X0, Y0, X-0, etc).
    """
    tiempo = None
    x_coord = None
    y_coord = None

    # Normalizar nombre sin extensión
    nombre = os.path.basename(str(nombre_archivo))
    nombre_sin_ext = re.sub(r'\.\w+$', '', nombre)

    # 1) Intentar extraer tiempo como último token separado por '_' (ej. ..._10)
    partes = nombre_sin_ext.split('_')
    if partes and partes[-1].isdigit():
        try:
            tiempo = int(partes[-1])
        except:
            tiempo = None

    # 2) Fallback: patrones T10s o T10
    if tiempo is None:
        tiempo_match = re.search(r"[Tt](\d+)\s*[sS]?$", nombre_sin_ext)
        if tiempo_match:
            tiempo = int(tiempo_match.group(1))

    # 3) Extraer X y Y con varios posibles formatos: X_1, X1, X=1, X-1
    x_match = re.search(r"[Xx][_\-=]?(-?\d+)", nombre_sin_ext)
    if x_match:
        try:
            x_coord = int(x_match.group(1))
        except:
            x_coord = None

    y_match = re.search(r"[Yy][_\-=]?(-?\d+)", nombre_sin_ext)
    if y_match:
        try:
            y_coord = int(y_match.group(1))
        except:
            y_coord = None

    # 4) Si aún no encontró mediante regex, intentar buscar tokens tipo X<number> o Y<number>
    if x_coord is None:
        m = re.search(r"[Xx]\s*(\d+)", nombre_sin_ext)
        if m:
            x_coord = int(m.group(1))
    if y_coord is None:
        m = re.search(r"[Yy]\s*(\d+)", nombre_sin_ext)
        if m:
            y_coord = int(m.group(1))

    return tiempo, x_coord, y_coord

def normalizar_nombre_sensor(sensor_text):
    """Normaliza una entrada de cabecera de sensor a 'Presion-Sensor N' (N entero global).
       Acepta formatos como:
         - 'Presion-Sensor_0_1' -> 'Presion-Sensor 1'
         - 'Presion-Sensor_1_12' -> 'Presion-Sensor 24'
         - 'Presion-Sensor 5' -> 'Presion-Sensor 5'
       Devuelve None si no puede normalizar.
    """
    if pd.isna(sensor_text):
        return None
    s = str(sensor_text).strip()
    if not s:
        return None

    # Caso 'Presion-Sensor_offset_index' (ej 'Presion-Sensor_1_3')
    m = re.search(r'(?i)presion[-_ ]*sensor[_\-]?(\d+)[_\-](\d+)', s)
    if m:
        offset = int(m.group(1))
        idx = int(m.group(2))
        sensor_global = offset * 12 + idx
        return f"Presion-Sensor {sensor_global}"

    # Caso 'Presion-Sensor 12' o 'Presion-Sensor_12'
    m2 = re.search(r'(?i)presion[-_ ]*sensor[_\-\s]*(\d+)', s)
    if m2:
        sensor_global = int(m2.group(1))
        return f"Presion-Sensor {sensor_global}"

    # Caso donde venga solo un número al final
    nums = re.findall(r'(\d+)', s)
    if nums:
        # si hay dos números, considerar que puede ser offset,index
        if len(nums) >= 2:
            offset = int(nums[-2])
            idx = int(nums[-1])
            if 0 <= offset <= 9 and 1 <= idx <= 12:
                sensor_global = offset * 12 + idx
                return f"Presion-Sensor {sensor_global}"
        # si hay uno solo, usarlo como número de sensor
        sensor_global = int(nums[-1])
        return f"Presion-Sensor {sensor_global}"

    # Si no se reconoce, devolver la cadena original (por si acaso)
    return s


def obtener_numero_sensor_desde_columna(col_name):
    """Devuelve el número entero del sensor si el nombre de columna tiene 'Presion-Sensor N' (o similar), sino None."""
    if pd.isna(col_name):
        return None
    s = str(col_name)
    m = re.search(r'(?i)presion[-_ ]*sensor[_\-\s]*(\d+)', s)
    if m:
        return int(m.group(1))
    # si no coincide, intentar extraer último número
    nums = re.findall(r'(\d+)', s)
    if nums:
        return int(nums[-1])
    return None


def sensor_num_a_altura(sensor_num, y_traverser, posicion_inicial, distancia_entre_tomas, n_sensores, orden="asc"):
    if sensor_num is None:
        return None
    
    toma_index = int(sensor_num)  # ahora global: 1..21

    if orden == "asc":
        z_total = y_traverser + (toma_index - 1) * distancia_entre_tomas
    else:
        z_total = y_traverser + (n_sensores - toma_index) * distancia_entre_tomas


    return z_total

def extraer_nombre_base_archivo(nombre_archivo):
    """Extraer nombre base del archivo (sin extensión y sin 'incertidumbre_')"""
    nombre_base = nombre_archivo.replace('.csv', '').replace('incertidumbre_', '').replace('_', ' ')
    # Capitalizar primera letra de cada palabra
    return ' '.join(word.capitalize() for word in nombre_base.split())

def procesar_promedios(archivo_csv, orden="asc"):
    """Procesar archivo de incertidumbre y detectar automáticamente la cantidad de sensores."""
    try:
        df_raw = pd.read_csv(archivo_csv, sep=";", header=None, dtype=str)  # leer como texto para robustez

        # Buscar la palabra "importante" para determinar dónde terminar
        index_final = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("importante", case=False).any(), axis=1)].index
        if not index_final.empty:
            df_raw = df_raw.iloc[:index_final[0]]

        resultados = []

        # Procesar bloques de 10 filas (misma lógica base)
        for i in range(0, df_raw.shape[0], 10):
            bloque = df_raw.iloc[i:i+10]
            if bloque.empty or len(bloque) < 3:
                continue

            archivo = bloque.iloc[0, 0]
            raw_sensores = bloque.iloc[0, 1:].tolist()
            raw_valores = bloque.iloc[2, 1:].tolist()
            muestras = bloque.iloc[1, 1] if 1 < bloque.shape[1] else None

            # Normalizar: si la cabecera vino entera en una sola celda separada por ';'
            sensores_lista = []
            for entry in raw_sensores:
                if pd.isna(entry):
                    continue
                s = str(entry).strip()
                if ';' in s:
                    partes = [p.strip() for p in s.split(';') if p.strip()]
                    sensores_lista.extend(partes)
                else:
                    sensores_lista.append(s)

            # Lo mismo para valores: si hay una celda con ;, expandir
            valores_lista = []
            for entry in raw_valores:
                if pd.isna(entry):
                    continue
                s = str(entry).strip()
                if ';' in s:
                    partes = [p.strip() for p in s.split(';')]
                    valores_lista.extend(partes)
                else:
                    valores_lista.append(s)

            # Si por alguna razón no se alinean en longitud, ajustar
            n = max(len(sensores_lista), len(valores_lista))
            sensores_lista = (sensores_lista + [None] * n)[:n]
            valores_lista = (valores_lista + [None] * n)[:n]

            fila = {
                "Archivo": archivo,
                "Numero de muestras": muestras,
            }

            # Mapear cada sensor raw -> nombre normalizado y poner su valor
            for sensor_raw, valor_raw in zip(sensores_lista, valores_lista):
                nombre_sensor_norm = normalizar_nombre_sensor(sensor_raw)
                if nombre_sensor_norm is None:
                    continue
                # limpiar y convertir valor (coma -> punto)
                valor = valor_raw
                if isinstance(valor, str):
                    valor = valor.replace(',', '.').strip()
                try:
                    valor_num = float(valor) if (valor is not None and str(valor) != '') else np.nan
                except:
                    valor_num = np.nan

                fila[nombre_sensor_norm] = valor_num

            resultados.append(fila)

        df_resultado = pd.DataFrame(resultados)

        # Extraer tiempo y coordenadas desde nombre de archivo (columna "Archivo")
        if "Archivo" in df_resultado.columns:
            coordenadas_tiempo = df_resultado["Archivo"].apply(extraer_tiempo_y_coordenadas)
            df_resultado["Tiempo (s)"] = [coord[0] for coord in coordenadas_tiempo]
            df_resultado["X_coord"] = [coord[1] for coord in coordenadas_tiempo]
            df_resultado["Y_coord"] = [coord[2] for coord in coordenadas_tiempo]
        else:
            df_resultado["Tiempo (s)"] = None
            df_resultado["X_coord"] = None
            df_resultado["Y_coord"] = None

        # 🔎 Detectar cantidad de sensores automáticamente
        sensores_cols = [c for c in df_resultado.columns if re.search(r'Presion[-_ ]*Sensor', str(c), re.IGNORECASE)]
        if sensores_cols:
            n_sensores = max([obtener_numero_sensor_desde_columna(c) for c in sensores_cols if obtener_numero_sensor_desde_columna(c) is not None])
        else:
            n_sensores = 0

        # Guardar en atributos del DataFrame para usar después
        df_resultado.attrs["n_sensores"] = n_sensores

        return df_resultado

    except Exception as e:
        st.error(f"Error al procesar archivo: {str(e)}")
        return None



def crear_archivos_individuales_por_tiempo_y_posicion(df_resultado, nombre_archivo_fuente):
    """
    Crea sub-archivos usando el nombre original del CSV como identificador único
    para evitar colisiones cuando varios CSV tienen nombres 'normalizados' iguales.
    """
    sub_archivos = {}
    nombre_base = extraer_nombre_base_archivo(nombre_archivo_fuente)
    nombre_original = os.path.splitext(os.path.basename(nombre_archivo_fuente))[0]  # sin extensión

    for x_valor in sorted(df_resultado["X_coord"].dropna().unique()):
        df_x = df_resultado[df_resultado["X_coord"] == x_valor]

        for tiempo in sorted(df_x["Tiempo (s)"].dropna().unique()):
            df_xt = df_x[df_x["Tiempo (s)"] == tiempo]

            # Usar el nombre original en la clave para que no colisione con otros CSV
            clave_sub_archivo = f"{nombre_original}_X{x_valor}_T{tiempo}s"

            sub_archivos[clave_sub_archivo] = {
                'archivo_fuente': nombre_base,          # nombre legible/normalizado
                'archivo_origen': nombre_original,     # IDENTIFICADOR ÚNICO (nombre del archivo subido)
                'tiempo': tiempo,
                'x_traverser': x_valor,
                'datos': df_xt,
                'nombre_archivo': f"{nombre_original}_X{x_valor}_T{tiempo}s.csv",
                'num_posiciones_y': len(df_xt['Y_coord'].unique()) if 'Y_coord' in df_xt.columns else 1
            }

    return sub_archivos

def calcular_posiciones_sensores(distancia_toma_12, distancia_entre_tomas, n_sensores, orden="asc"):
    """
    Calcula las posiciones físicas de todos los sensores en función de:
    - distancia_toma_12: posición de la toma física número 12 (en mm)
    - distancia_entre_tomas: separación entre sensores consecutivos (en mm)
    - n_sensores: cantidad total de sensores detectados en el archivo
    - orden: "asc" o "des" (según cómo están montados los sensores)
    Devuelve un diccionario con la posición y número físico de cada sensor.
    """
    posiciones = {}
    for sensor_num in range(1, n_sensores + 1):
        if orden == "asc":
            y_position = (sensor_num - 1) * distancia_entre_tomas
        else:
            y_position = (n_sensores - sensor_num) * distancia_entre_tomas


        posiciones[f"Presion-Sensor {sensor_num}"] = {
            'x': 0,
            'y': y_position,
            'sensor_fisico': sensor_num
        }
    return posiciones


def crear_grafico_betz_concatenado(sub_archivos_seleccionados, posiciones_sensores, configuracion):
    fig = go.Figure()

    posicion_inicial = configuracion['distancia_toma_12']
    distancia_entre_tomas = configuracion['distancia_entre_tomas']
    orden = configuracion.get('orden', 'asc')
    colores_por_tiempo = {10: '#08596C', 20: '#E74C3C', 30: '#F39C12', 40: '#27AE60', 50: '#8E44AD', 60: '#3498DB'}

    datos_agrupados = {}
    for clave, sub_archivo in sub_archivos_seleccionados.items():
        grupo = (sub_archivo['archivo_fuente'], sub_archivo['tiempo'])
        datos_agrupados.setdefault(grupo, []).append(sub_archivo)

    for grupo, sub_archivos_del_grupo in datos_agrupados.items():
        archivo_fuente, tiempo = grupo
        color = colores_por_tiempo.get(tiempo, '#333333')

        z_grupo, presion_grupo = [], []

        for sub_archivo in sub_archivos_del_grupo:
            datos_tiempo = sub_archivo['datos']
            sensor_cols = [c for c in datos_tiempo.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

            for _, fila in datos_tiempo.iterrows():
                y_traverser = fila.get('Y_coord', 0) if pd.notna(fila.get('Y_coord', np.nan)) else 0

                for col in sensor_cols:
                    sensor_num = obtener_numero_sensor_desde_columna(col)
                    if sensor_num is None:
                        continue
                    z_total = sensor_num_a_altura(sensor_num, y_traverser, posicion_inicial, distancia_entre_tomas, configuracion.get('n_sensores', len(sensor_cols)), orden)
                    presion = fila.get(col, None)
                    if pd.isna(presion):
                        continue
                    try:
                        presion = float(str(presion).replace(',', '.'))
                        presion_grupo.append(presion)
                        z_grupo.append(z_total)
                    except ValueError:
                        continue


def extraer_datos_para_grafico(sub_archivo, configuracion):
    """Extraer datos de presión y altura de un sub-archivo para gráficos (múltiples posiciones).
       Ahora soporta sensores numerados dinámicamente.
    """
    datos_tiempo = sub_archivo['datos']
    distancia_entre_tomas = configuracion['distancia_entre_tomas']
    posicion_inicial = configuracion.get('distancia_toma_12', 0)
    orden = configuracion.get('orden', 'asc')

    z_datos, presion_datos = [], []

    sensor_cols = [c for c in datos_tiempo.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
    n_sensores = max([obtener_numero_sensor_desde_columna(c) for c in sensor_cols], default=0)

    for _, fila in datos_tiempo.iterrows():
        y_traverser = fila.get('Y_coord', 0) if pd.notna(fila.get('Y_coord', np.nan)) else 0

        for col in sensor_cols:
            sensor_num = obtener_numero_sensor_desde_columna(col)
            if sensor_num is None:
                continue
            z_total = sensor_num_a_altura(sensor_num, y_traverser, posicion_inicial, distancia_entre_tomas, n_sensores, orden)
            presion = fila.get(col, None)
            if pd.isna(presion):
                continue
            try:
                presion_val = float(str(presion).replace(',', '.'))
                z_datos.append(z_total)
                presion_datos.append(presion_val)
            except (ValueError, TypeError):
                continue

    # Ordenar y devolver
    if z_datos and presion_datos:
        datos_ordenados = sorted(zip(z_datos, presion_datos))
        z_ordenado, presion_ordenada = zip(*datos_ordenados)
        return list(z_ordenado), list(presion_ordenada)

    # 🔑 SIEMPRE devolver dos listas
    return [], []


def calcular_area_bajo_curva(z_datos, presion_datos):
    """Calcular área bajo la curva usando regla del trapecio"""
    if len(z_datos) < 2 or len(presion_datos) < 2:
        return 0
    
    area = 0
    for i in range(len(z_datos) - 1):
        # Regla del trapecio
        h = z_datos[i + 1] - z_datos[i]
        area += h * (presion_datos[i] + presion_datos[i + 1]) / 2
    
    return abs(area)

def crear_superficie_diferencia_delaunay_3d(datos_a, datos_b, nombre_a, nombre_b, configuracion_3d, mostrar_puntos=True):
    """
    Crea una superficie 3D de diferencias con Delaunay y mejoras visuales.
    """
    try:
        posicion_inicial = configuracion_3d['distancia_toma_12']
        distancia_entre_tomas = configuracion_3d['distancia_entre_tomas']
        orden = configuracion_3d['orden']

        def extraer_puntos(datos):
            puntos = {}
            sensor_cols = [c for c in datos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
            for _, fila in datos.iterrows():
                x_traverser = fila.get('X_coord', None)
                y_traverser = fila.get('Y_coord', None)
                if pd.isna(x_traverser) or pd.isna(y_traverser):
                    continue
                for col in sensor_cols:
                    sensor_num = obtener_numero_sensor_desde_columna(col)
                    if sensor_num is None:
                        continue
                    altura_sensor_real = sensor_num_a_altura(
                        sensor_num, y_traverser, posicion_inicial, distancia_entre_tomas, orden
                    )
                    presion = fila.get(col, None)
                    if pd.isna(presion) or presion is None:
                        continue
                    try:
                        if isinstance(presion, str):
                            presion = float(presion.replace(',', '.'))
                        puntos[(x_traverser, altura_sensor_real)] = float(presion)
                    except (ValueError, TypeError):
                        continue
            return puntos

        puntos_a = extraer_puntos(datos_a)
        puntos_b = extraer_puntos(datos_b)

        puntos_comunes = set(puntos_a.keys()) & set(puntos_b.keys())
        if len(puntos_comunes) < 4:
            st.error("No hay suficientes puntos comunes para generar la resta de superficies.")
            return None

        puntos_x, puntos_y, puntos_z = [], [], []
        for (x, y) in puntos_comunes:
            diff = puntos_a[(x, y)] - puntos_b[(x, y)]
            puntos_x.append(x)
            puntos_y.append(y)
            puntos_z.append(diff)

        puntos_2d = np.vstack([puntos_x, puntos_y]).T
        tri = Delaunay(puntos_2d)

        fig = go.Figure()

        # Superficie
        fig.add_trace(go.Mesh3d(
            x=puntos_x,
            y=puntos_y,
            z=puntos_z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=puntos_z,
            colorscale='Turbo',
            colorbar_title='Δ Presión [Pa]',
            name=f"Diferencia {nombre_a} - {nombre_b}",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=100),
            hovertemplate='<b>Δ Presión</b>: %{intensity:.3f} Pa<br>Pos X: %{x:.1f} mm<br>Altura: %{y:.1f} mm<extra></extra>'
        ))

        # Wireframe
        wire_x, wire_y, wire_z = [], [], []
        for simplex in tri.simplices:
            for idx_pair in [(0,1), (1,2), (2,0)]:
                for idx in idx_pair:
                    wire_x.append(puntos_x[simplex[idx]])
                    wire_y.append(puntos_y[simplex[idx]])
                    wire_z.append(puntos_z[simplex[idx]])
                wire_x.append(None)
                wire_y.append(None)
                wire_z.append(None)

        fig.add_trace(go.Scatter3d(
            x=wire_x,
            y=wire_y,
            z=wire_z,
            mode='lines',
            line=dict(color='black', width=1),
            name='Malla',
            hoverinfo='skip'
        ))

        # Puntos medidos
        if mostrar_puntos:
            fig.add_trace(go.Scatter3d(
                x=puntos_x,
                y=puntos_y,
                z=puntos_z,
                mode='markers',
                marker=dict(size=3, color='red'),
                name='Puntos medidos',
                hovertemplate='<b>Punto medido</b><br>Δ Presión: %{z:.3f} Pa<br>Pos X: %{x:.1f} mm<br>Altura: %{y:.1f} mm<extra></extra>'
            ))

        fig.update_layout(
            title=f"Diferencia de Superficies 3D Mejorada - {nombre_a} - {nombre_b}",
            scene=dict(
                xaxis_title="Posición X Traverser [mm]",
                yaxis_title="Altura Física Z [mm]",
                zaxis_title="Δ Presión [Pa]",
                aspectratio=dict(x=1.5, y=2, z=1),
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            ),
            width=1600,
            height=900,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        return fig

    except Exception as e:
        st.error(f"Error creando la superficie de diferencia 3D mejorada: {str(e)}")
        return None




def crear_superficie_diferencia(datos_a, datos_b, nombre_a, nombre_b):
    """
    Resta dos superficies 3D: para cada (X,Y) común calcula la media de
    todas las columnas 'Presion-Sensor N' presentes en esa fila y resta.
    """
    coords_a = set(tuple(row) for row in datos_a[['X_coord', 'Y_coord']].dropna().to_numpy())
    coords_b = set(tuple(row) for row in datos_b[['X_coord', 'Y_coord']].dropna().to_numpy())
    coords_comunes = sorted(list(coords_a & coords_b))

    if len(coords_comunes) < 4:
        st.warning("No hay suficientes puntos (X,Y) en común para generar una superficie.")
        return None

    X_final, Y_final, Z_final = [], [], []

    for x_coord, y_coord in coords_comunes:
        fila_a = datos_a[(datos_a['X_coord'] == x_coord) & (datos_a['Y_coord'] == y_coord)]
        fila_b = datos_b[(datos_b['X_coord'] == x_coord) & (datos_b['Y_coord'] == y_coord)]

        if fila_a.empty or fila_b.empty:
            continue

        # detectar columnas de sensores en cada fila y promediar
        cols_a = [c for c in fila_a.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]
        cols_b = [c for c in fila_b.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

        presiones_a = []
        for c in cols_a:
            try:
                val = fila_a.iloc[0][c]
                if isinstance(val, str):
                    val = float(val.replace(',', '.'))
                presiones_a.append(float(val))
            except:
                continue

        presiones_b = []
        for c in cols_b:
            try:
                val = fila_b.iloc[0][c]
                if isinstance(val, str):
                    val = float(val.replace(',', '.'))
                presiones_b.append(float(val))
            except:
                continue

        if presiones_a and presiones_b:
            diferencia = np.mean(presiones_a) - np.mean(presiones_b)
            X_final.append(x_coord)
            Y_final.append(y_coord)
            Z_final.append(diferencia)

    if len(X_final) < 4:
        st.warning("No se pudieron generar suficientes puntos de diferencia para la superficie.")
        return None

    # Crear la malla
    X_unique = sorted(list(set(X_final)))
    Y_unique = sorted(list(set(Y_final)))
    Z_matrix = np.full((len(Y_unique), len(X_unique)), np.nan)

    for x, y, z in zip(X_final, Y_final, Z_final):
        iy = Y_unique.index(y)
        ix = X_unique.index(x)
        Z_matrix[iy, ix] = z

    X_mesh, Y_mesh = np.meshgrid(X_unique, Y_unique)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X_mesh, y=Y_mesh, z=Z_matrix,
        colorscale='RdBu_r',
        colorbar=dict(title="Diferencia de Presión [Pa]"),
        hovertemplate='<b>Diferencia de Presión</b><br>X: %{x:.1f} mm, Y: %{y:.1f} mm<br>Diferencia: %{z:.3f} Pa<extra></extra>'
    ))

    fig.update_layout(
        title=f"Diferencia de Superficies: {nombre_a} vs {nombre_b}",
        scene=dict(
            xaxis_title="Posición X [mm]",
            yaxis_title="Posición Y [mm]",
            zaxis_title="Diferencia de Presión [Pa]"
        ),
        font=dict(color="black")
    )
    fig.update_layout(width=1600, height=900, margin=dict(l=0, r=0, t=50, b=0))
    return fig

    
def crear_grafico_diferencia_areas(sub_archivo_a, sub_archivo_b, configuracion):
    """Crear gráfico mostrando la diferencia como UNA sola área"""
    
    # Extraer datos de ambos sub-archivos
    z_a, presion_a = extraer_datos_para_grafico(sub_archivo_a, configuracion)
    z_b, presion_b = extraer_datos_para_grafico(sub_archivo_b, configuracion)
    
    if not z_a or not z_b or not presion_a or not presion_b:
        return None, 0
    
    # Crear gráfico
    fig = go.Figure()
    
    # Agregar líneas de referencia (más tenues)
    fig.add_trace(go.Scatter(
        x=presion_a, y=z_a,
        mode='lines',
        name=f"{sub_archivo_a['archivo_fuente']} T{sub_archivo_a['tiempo']}s",
        line=dict(color='#08596C', width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                    'Presión: %{x:.3f} Pa<br>' +
                    'Altura: %{y:.1f} mm<br>' +
                    '<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=presion_b, y=z_b,
        mode='lines',
        name=f"{sub_archivo_b['archivo_fuente']} T{sub_archivo_b['tiempo']}s",
        line=dict(color='#E74C3C', width=2, dash='dot'),
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                    'Presión: %{x:.3f} Pa<br>' +
                    'Altura: %{y:.1f} mm<br>' +
                    '<extra></extra>'
    ))
    
    # Calcular diferencia punto a punto (interpolando si es necesario)
    # Usar el rango de alturas común
    z_min = max(min(z_a), min(z_b))
    z_max = min(max(z_a), max(z_b))
    
    # Crear puntos interpolados
    z_interp = np.linspace(z_min, z_max, 50)
    
    # Interpolar presiones
    presion_a_interp = np.interp(z_interp, z_a, presion_a)
    presion_b_interp = np.interp(z_interp, z_b, presion_b)
    
    # Calcular diferencia
    diferencia_presion = presion_a_interp - presion_b_interp
    
    # Crear área de diferencia ÚNICA
    # Determinar color basado en si la diferencia es mayormente positiva o negativa
    diferencia_promedio = np.mean(diferencia_presion)
    color_diferencia = '#27AE60' if diferencia_promedio >= 0 else '#E67E22'  # Verde si A>B, naranja si B>A
    
    # Crear área desde cero hasta la diferencia
    x_area = [0] + list(diferencia_presion) + [0]
    y_area = [z_interp[0]] + list(z_interp) + [z_interp[-1]]
    
    fig.add_trace(go.Scatter(
        x=x_area, y=y_area,
        fill='toself',
        fillcolor=f'rgba({int(color_diferencia[1:3], 16)}, {int(color_diferencia[3:5], 16)}, {int(color_diferencia[5:7], 16)}, 0.4)',
        line=dict(color=color_diferencia, width=3),
        name=f'Diferencia: {sub_archivo_a["archivo_fuente"]} - {sub_archivo_b["archivo_fuente"]}',
        hovertemplate='<b>Diferencia</b><br>' +
                    'Diferencia: %{x:.3f} Pa<br>' +
                    'Altura: %{y:.1f} mm<br>' +
                    '<extra></extra>'
    ))
    
    # Calcular área total de diferencia
    area_diferencia = np.trapz(np.abs(diferencia_presion), z_interp)
    
    # Layout CON LEYENDA MEJORADA
    fig.update_layout(
        title=f"Diferencia de Perfiles: {sub_archivo_a['archivo_fuente']} - {sub_archivo_b['archivo_fuente']}",
        xaxis_title="Presión / Diferencia de Presión [Pa]",
        yaxis_title="Altura z [mm]",
        height=700, width=1000,  # Más ancho para leyenda
        showlegend=True,  # FORZAR LEYENDA VISIBLE
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color="#FFFFFF"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#000005",
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2,
            scaleanchor="y",      # AGREGADO: Configuración solicitada
            scaleratio=4          # AGREGADO: Configuración solicitada
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#080000",
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#08596C',
            borderwidth=2,
            x=1.02,
            y=1,
            font=dict(size=12, color='black')  # AGREGAR COLOR NEGRO
        )
    )
    fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',   # Fondo transparente
    paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
    font=dict(color='white'),       # Texto en blanco
    xaxis_title="Presión Total [Pa]",
    yaxis_title="Altura Z [mm]",
    height=900,
    width=1600
    )
    return fig, area_diferencia

def mostrar_configuracion_sensores(section_key):
    """Muestra los widgets de configuración de sensores y guarda el estado."""
    st.markdown("### 📍 Configuración de Sensores y Geometría")

    config_key = f'configuracion_{section_key}'
    if config_key not in st.session_state:
        st.session_state[config_key] = {}

    # Orden de sensores
    orden_sensores = st.selectbox(
        "Orden de lectura de sensores:", ["asc", "des"],
        format_func=lambda x: "Ascendente (sensor 1 abajo, 12 arriba)" if x == "asc" else "Descendente (sensor 12 abajo, 1 arriba)",
        help="Define cómo se leen los datos de los sensores en relación a su posición física.",
        key=f'orden_sensores_{section_key}'
    )
    
    st.info("🔍 **Pregunta:** ¿Qué sensor corresponde a la toma número 12 (la que se encuentra cerca del piso)?")
    sensor_referencia = st.selectbox(
        "Sensor de referencia (toma 12):", [f"Sensor {i}" for i in range(1, 13)],
        index=11, help="Seleccione el sensor que corresponde a la toma física número 12.",
        key=f'sensor_ref_{section_key}'
    )
    
    distancia_toma_12 = st.number_input(
        "Distancia de la toma 12 a la posición X=0, Y=0 del traverser [mm]:",
        value=-120.0, step=1.0, format="%.1f",
        help="Distancia en mm desde el punto de referencia del traverser.",
        key=f'dist_toma_{section_key}'
    )
    
    distancia_entre_tomas = st.number_input(
        "Distancia entre tomas [mm]:", value=10.91, step=0.01, format="%.2f",
        help="Distancia física entre tomas consecutivas según el plano técnico.",
        key=f'dist_entre_{section_key}'
    )
    
    if st.button(f"💾 Guardar Configuración", type="primary", key=f'save_config_{section_key}'):
        st.session_state[config_key] = {
            'orden': orden_sensores,
            'sensor_referencia': sensor_referencia,
            'distancia_toma_12': distancia_toma_12,
            'distancia_entre_tomas': distancia_entre_tomas
        }
        st.success(f"✅ Configuración para la sección {section_key.upper()} guardada.")
        st.rerun()

    return st.session_state.get(config_key, {})

def crear_superficie_delaunay_3d(datos_completos, configuracion_3d, nombre_archivo, mostrar_puntos=True):
    """
    Crea una superficie 3D continua con Delaunay y mejoras visuales.
    Ahora permite activar/desactivar la visualización de puntos medidos.
    """
    try:
        posicion_inicial = configuracion_3d['distancia_toma_12']
        distancia_entre_tomas = configuracion_3d['distancia_entre_tomas']
        orden = configuracion_3d['orden']

        puntos_x, puntos_y_altura, presiones_z = [], [], []

        # Detectar columnas de sensores
        sensor_cols = [c for c in datos_completos.columns if re.search(r'(?i)presion[-_ ]*sensor', str(c))]

        for _, fila in datos_completos.iterrows():
            x_traverser = fila.get('X_coord', None)
            y_traverser = fila.get('Y_coord', None)
            if pd.isna(x_traverser) or pd.isna(y_traverser):
                continue

            for col in sensor_cols:
                sensor_num = obtener_numero_sensor_desde_columna(col)
                if sensor_num is None:
                    continue
                altura_sensor_real = sensor_num_a_altura(
                    sensor_num, y_traverser, posicion_inicial, distancia_entre_tomas, orden
                )
                presion = fila.get(col, None)
                if pd.isna(presion) or presion is None:
                    continue
                try:
                    if isinstance(presion, str):
                        presion = float(presion.replace(',', '.'))
                    presion_val = float(presion)
                    puntos_x.append(x_traverser)
                    puntos_y_altura.append(altura_sensor_real)
                    presiones_z.append(presion_val)
                except (ValueError, TypeError):
                    continue

        if len(puntos_x) < 4:
            st.error("No hay suficientes datos válidos para generar una superficie.")
            return None

        # Triangulación Delaunay
        puntos_2d = np.vstack([puntos_x, puntos_y_altura]).T
        tri = Delaunay(puntos_2d)

        fig = go.Figure()

        # Superficie principal
        fig.add_trace(go.Mesh3d(
            x=puntos_x,
            y=puntos_y_altura,
            z=presiones_z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=presiones_z,
            colorscale='Turbo',
            colorbar_title='Presión [Pa]',
            name='Superficie de presión',
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5, fresnel=0.2),
            lightposition=dict(x=100, y=200, z=100),
            hovertemplate='<b>Presión</b>: %{intensity:.3f} Pa<br>Pos X: %{x:.1f} mm<br>Altura: %{y:.1f} mm<extra></extra>'
        ))

        # Wireframe
        wire_x, wire_y, wire_z = [], [], []
        for simplex in tri.simplices:
            for idx_pair in [(0,1), (1,2), (2,0)]:
                for idx in idx_pair:
                    wire_x.append(puntos_x[simplex[idx]])
                    wire_y.append(puntos_y_altura[simplex[idx]])
                    wire_z.append(presiones_z[simplex[idx]])
                wire_x.append(None)
                wire_y.append(None)
                wire_z.append(None)

        fig.add_trace(go.Scatter3d(
            x=wire_x,
            y=wire_y,
            z=wire_z,
            mode='lines',
            line=dict(color='black', width=1),
            name='Malla',
            hoverinfo='skip'
        ))

        # Puntos medidos (si mostrar_puntos=True)
        if mostrar_puntos:
            fig.add_trace(go.Scatter3d(
                x=puntos_x,
                y=puntos_y_altura,
                z=presiones_z,
                mode='markers',
                marker=dict(size=3, color='red'),
                name='Puntos medidos',
                hovertemplate='<b>Punto medido</b><br>Presión: %{z:.3f} Pa<br>Pos X: %{x:.1f} mm<br>Altura: %{y:.1f} mm<extra></extra>'
            ))

        fig.update_layout(
            title=f"Superficie de Presión 3D Mejorada - {nombre_archivo}",
            scene=dict(
                xaxis_title="Posición X Traverser [mm]",
                yaxis_title="Altura Física Real del Sensor [mm]",
                zaxis_title="Presión [Pa]",
                aspectratio=dict(x=1, y=2, z=0.8),
                camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
            ),
            width=1600,
            height=900,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        return fig

    except Exception as e:
        st.error(f"Error creando la superficie de malla 3D mejorada: {str(e)}")
        return None



    
def crear_sub_archivos_3d_por_tiempo_y_posicion(df_datos, nombre_archivo):
    """Crear sub-archivos 3D por tiempo y posición (similar a 2D)"""
    sub_archivos = {}
    
    # Obtener tiempos únicos
    tiempos_unicos = df_datos["Tiempo (s)"].dropna().unique()
    
    for tiempo in tiempos_unicos:
        # Filtrar datos por tiempo
        df_tiempo = df_datos[df_datos["Tiempo (s)"] == tiempo].copy()
        
        # Crear clave para el sub-archivo
        clave_sub_archivo = f"{nombre_archivo}_T{tiempo}s"

        sub_archivos[clave_sub_archivo] = {
            'archivo_fuente': nombre_archivo,
            'tiempo': tiempo,
            'datos': df_tiempo,
            'nombre_archivo': f"{nombre_archivo}_T{tiempo}s.csv"
        }
    
    return sub_archivos

def mostrar_resumen_archivos_tabla(sub_archivos_por_fuente):
    """Mostrar resumen de archivos en formato tabla organizada"""
    st.markdown("### 📊 Resumen de Sub-archivos Generados")
    
    # Crear datos para la tabla
    datos_tabla = []
    
    for archivo_fuente, tiempos_dict in sub_archivos_por_fuente.items():
        for tiempo, sub_archivos_tiempo in tiempos_dict.items():
            for clave, sub_archivo in sub_archivos_tiempo:
                datos_tabla.append({
                    'Archivo_Fuente': archivo_fuente,
                    'Tiempo_s': f"T{tiempo}s", 
                    'Posición_X': sub_archivo['x_traverser'],
                    'Registros': len(sub_archivo['datos']),
                    'Nombre_Archivo': sub_archivo['nombre_archivo'],
                    'Clave': clave
                })
    
    # Crear DataFrame y mostrar como tabla
    df_resumen = pd.DataFrame(datos_tabla).sort_values(['Archivo_Fuente', 'Posición_X', 'Tiempo_s'])
    
    # NUEVO: Mostrar tabla con separadores correctos para CSV
    st.dataframe(
        df_resumen[['Archivo_Fuente', 'Tiempo_s', 'Registros', 'Nombre_Archivo']], 
        use_container_width=True,
        hide_index=True
    )
    
    # NUEVO: Botón para descargar tabla como CSV bien formateado
    csv_tabla = df_resumen[['Archivo_Fuente', 'Tiempo_s', 'Registros', 'Nombre_Archivo']].to_csv(
        index=False, 
        sep=';',  # CAMBIAR a punto y coma para Excel
        encoding='utf-8-sig',  # CAMBIAR encoding para Excel
        decimal=','  # AGREGAR separador decimal para Excel
    )
    
    st.download_button(
        label="📥 Descargar Tabla Resumen (CSV)",
        data=csv_tabla,
        file_name=f"resumen_subarchivos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Estadísticas adicionales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sub-archivos", len(datos_tabla))
    with col2:
        st.metric("Archivos Fuente", len(sub_archivos_por_fuente))
    with col3:
        total_registros = sum([item['Registros'] for item in datos_tabla])
        st.metric("Total Registros", total_registros)

# Sidebar para navegación (colapsable)
with st.sidebar:
    st.markdown("### 🧭 Navegación")
    
    # --- CÓDIGO CORREGIDO ---
    # Estos botones ahora solo cambian de sección, sin borrar nada.
    
    if st.button("🏠 Inicio", use_container_width=True):
        st.session_state.seccion_actual = 'inicio'
        st.rerun()
    
    if st.button("📊 BETZ 2D", use_container_width=True):
        st.session_state.seccion_actual = 'betz_2d'
        st.rerun()

    if st.button("🌪️ BETZ 3D", use_container_width=True):
        st.session_state.seccion_actual = 'betz_3d'
        st.rerun()

    if st.button("🖥️ Visualización de Resultados", use_container_width=True):
        st.session_state.seccion_actual = 'visualizacion'
        st.rerun()
    
    st.divider()
    
    # El resto del código de la sidebar no necesita cambios
    st.markdown("### ℹ️ Información")
    st.markdown(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}")
    st.markdown(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.session_state.datos_procesados:
        st.markdown(f"**Archivos procesados:** {len(st.session_state.datos_procesados)}")
    
    if st.session_state.sub_archivos_generados:
        st.markdown(f"**Sub-archivos 2D:** {len(st.session_state.sub_archivos_generados)}")
    
    # Corregido para usar la clave correcta de las diferencias guardadas
    if st.session_state.get('diferencias_guardadas'):
        st.markdown(f"**Diferencias guardadas:** {len(st.session_state.diferencias_guardadas)}")
# Contenido principal según la sección
if st.session_state.seccion_actual == 'inicio':
    # Página de inicio
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            Laboratorio de Aerodinámica y Fluidos
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            UTN HAEDO
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Secciones principales
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3 style="color: #08596C; font-size: 2rem; margin-bottom: 1rem; text-align: center;">
                📊 BETZ 2D
            </h3>
            <p style="color: #4b5563; line-height: 1.6; margin-bottom: 1.5rem; text-align: center;">
                Análisis bidimensional de perfiles de presión.<br><br>
                • Procesamiento automático de archivos CSV<br>
                • Extracción de tiempo y coordenadas X-Y<br>
                • Gráficos en la seccion X <br>
                • Configuración flexible de sensores <br>
                • Análisis de Diferencias de Áreas
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ACCEDER A BETZ 2D", key="betz_2d_btn", type="primary", use_container_width=True):
            st.session_state.seccion_actual = 'betz_2d'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3 style="color: #08596C; font-size: 2rem; margin-bottom: 1rem; text-align: center;">
                🌪️ BETZ 3D
            </h3>
            <p style="color: #4b5563; line-height: 1.6; margin-bottom: 1.5rem; text-align: center;">
                Análisis tridimensional completo de flujos.<br><br>
                • Procesamiento automático de archivos CSV<br>
                • Configuración flexible de sensores <br>
                • Análisis de múltiples posiciones X, Y<br>
                • Visualización 3D de campos de presión<br>
                • Análisis de Diferencias de Superficies<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ACCEDER A BETZ 3D", key="betz_3d_btn", type="primary", use_container_width=True):
            st.session_state.seccion_actual = 'betz_3d'
            st.rerun()

elif st.session_state.seccion_actual == 'betz_2d':
    if st.session_state.configuracion_inicial:
        st.markdown("# 📊 BETZ 2D - Análisis Bidimensional")
        st.markdown("Análisis de perfiles de presión concatenados con extracción automática de tiempo y coordenadas")

    # --- Inicializar variables persistentes ---
    if "datos_procesados_betz2d" not in st.session_state:
        st.session_state.datos_procesados_betz2d = {}
    if "sub_archivos_betz2d" not in st.session_state:
        st.session_state.sub_archivos_betz2d = {}
    if "uploaded_files_betz2d" not in st.session_state:
        st.session_state.uploaded_files_betz2d = []

    # Paso 1: Configuración inicial
    st.markdown("## ⚙️ Paso 1: Configuración Inicial")

    # Reorganizar: datos a la izquierda, imagen más pequeña a la derecha
    col_datos, col_imagen = st.columns([2, 1])

    with col_datos:
        st.markdown("### 📍 Configuración de Sensores y Geometría")

        orden_sensores = st.selectbox(
            "Orden de lectura de sensores:",
            ["asc", "des"],
            format_func=lambda x: "Ascendente (sensor 1 más abajo al 12 más arriba)" if x == "asc" else "Descendente (sensor 12 más abajo y sensor 1 más arriba)",
            help="Define cómo se leen los datos de los sensores en relación a su posición física"
        )

        st.info("🔍 **Pregunta:** ¿Qué sensor corresponde a la toma número 12 (la que se encuentra cerca del piso)?")
        sensor_referencia = st.selectbox(
            "Sensor de referencia (toma 12):",
            [f"Sensor {i}" for i in range(1, 37)],  # ahora permite hasta 36
            index=11,
            help="Seleccione el sensor que corresponde a la toma física número 12"
        )

        distancia_toma_12 = st.number_input(
            "Distancia de la toma 12 a la posición X=0, Y=0 (coordenadas del traverser) [mm]:",
            value=-120.0,
            step=1.0,
            format="%.1f",
            help="Distancia en mm desde el punto de referencia del traverser"
        )

        distancia_entre_tomas = st.number_input(
            "Distancia entre tomas [mm]:",
            value=10.91,
            step=0.01,
            format="%.2f",
            help="Distancia física entre tomas consecutivas según el plano técnico"
        )

        if st.button("💾 Guardar Configuración", type="primary"):
            st.session_state.configuracion_inicial = {
                'orden': orden_sensores,
                'sensor_referencia': sensor_referencia,
                'distancia_toma_12': distancia_toma_12,
                'distancia_entre_tomas': distancia_entre_tomas
            }
            st.success("✅ Configuración guardada correctamente")
            st.rerun()

    with col_imagen:
        st.markdown("### 📐 Diagrama de Referencia")
        st.markdown("""
        <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2rem; text-align: center; color: #64748b;">
            <h4>📐 Diagrama de Referencia</h4>
            <img src="https://raw.githubusercontent.com/Juan-Cruz-de-la-Fuente/Laboratorio/main/Peine.jpg" 
                alt="Diagrama técnico" 
                style="max-width:100%; height:auto; border-radius: 8px;">
            <p><small>Subir imagen del plano técnico</small></p>
        </div>
        """, unsafe_allow_html=True)

    # Paso 2: Carga de archivos
    if st.session_state.configuracion_inicial:
        st.markdown("## 📁 Paso 2: Carga de Archivos de Incertidumbre")
        uploaded_files = st.file_uploader(
            "Seleccione uno o más archivos CSV de incertidumbre:",
            type=['csv'],
            accept_multiple_files=True,
            key="uploader_betz2d",
            help="Los archivos deben tener el formato estándar del laboratorio..."
        )

        if uploaded_files:
            st.session_state.uploaded_files_betz2d = uploaded_files
        else:
            uploaded_files = st.session_state.uploaded_files_betz2d

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s)")

        for archivo in uploaded_files:
            if archivo.name not in st.session_state.datos_procesados_betz2d:
                with st.spinner(f"Procesando {archivo.name}..."):
                    datos = procesar_promedios(archivo, st.session_state.configuracion_inicial['orden'])
                    if datos is not None:
                        st.session_state.datos_procesados_betz2d[archivo.name] = datos
                        st.session_state.datos_procesados[archivo.name] = datos
                        sub_archivos = crear_archivos_individuales_por_tiempo_y_posicion(datos, archivo.name)

                        if 'sub_archivos_generados' not in st.session_state:
                            st.session_state.sub_archivos_generados = {}

                        st.session_state.sub_archivos_generados.update(sub_archivos)

                        nombre_base = extraer_nombre_base_archivo(archivo.name)
                        datos_ordenados_nombre = datos.sort_values("Archivo")
                        datos_ordenados_x = datos.sort_values("X_coord")
                        datos_ordenados_tiempo = datos.sort_values("Tiempo (s)")

                        for label, df_ordenado in {
                            "ordenado_nombre": datos_ordenados_nombre,
                            "ordenado_x": datos_ordenados_x,
                            "ordenado_tiempo": datos_ordenados_tiempo
                        }.items():
                            csv_bytes = df_ordenado.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                            st.download_button(
                                label=f"📥 Descargar {nombre_base}_{label}.csv",
                                data=csv_bytes,
                                file_name=f"{nombre_base}_{label}.csv",
                                mime="text/csv"
                            )

                        st.success(f"✅ {archivo.name} procesado. Se generaron {len(sub_archivos)} sub-archivos.")


        # Mostrar datos procesados
        if st.session_state.datos_procesados:
            st.markdown("## 📋 Datos Procesados")
            for nombre_archivo, datos in st.session_state.datos_procesados.items():
                with st.expander(f"Ver datos de {nombre_archivo}"):
                    st.dataframe(datos, use_container_width=True)
                    tiempos_unicos = datos['Tiempo (s)'].dropna().unique()
                    coordenadas_unicas = datos[['X_coord', 'Y_coord']].drop_duplicates()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Tiempos encontrados:** {sorted(tiempos_unicos)}")
                    with col2:
                        st.markdown("**Coordenadas encontradas:**")
                        st.dataframe(coordenadas_unicas, use_container_width=True)


        st.header("📂 Paso 3: Sub-archivos Generados por Archivo y Tiempo")

        # --- 1) Recolectar todos los sub-archivos desde session_state ---
        filas_resumen = []
        for clave, sub in st.session_state.sub_archivos_generados.items():
            archivo_origen = sub.get('archivo_origen', sub.get('archivo_fuente', 'SinOrigen'))
            tiempo = sub.get('tiempo', 'SinTiempo')
            x_trav = sub.get('x_traverser', None)
            registros = len(sub.get('datos', pd.DataFrame()))
            nombre_archivo = sub.get('nombre_archivo', f"{clave}.csv")
            filas_resumen.append({
                'Archivo_Fuente': archivo_origen,
                'Tiempo_s': f"T{tiempo}s" if pd.notna(tiempo) else 'SinTiempo',
                'Posicion_X': x_trav,
                'Registros': registros,
                'Nombre_Archivo': nombre_archivo,
                'Clave': clave
            })

        # --- 2) DataFrame resumen y descarga ---
        if filas_resumen:
            df_resumen = pd.DataFrame(filas_resumen)
            # ordenar para mejor lectura
            df_resumen = df_resumen.sort_values(['Archivo_Fuente', 'Tiempo_s', 'Posicion_X'], na_position='last').reset_index(drop=True)

            st.markdown("### 📊 Resumen de Sub-archivos Generados")
            st.dataframe(df_resumen[['Archivo_Fuente', 'Tiempo_s', 'Posicion_X', 'Registros', 'Nombre_Archivo']], use_container_width=True)

            csv_resumen = df_resumen[['Archivo_Fuente', 'Tiempo_s', 'Posicion_X', 'Registros', 'Nombre_Archivo']].to_csv(
                index=False, sep=';', encoding='utf-8-sig', decimal=','
            ).encode('utf-8-sig')

            st.download_button(
                label="📥 Descargar Tabla Resumen (CSV)",
                data=csv_resumen,
                file_name=f"resumen_subarchivos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"desc_resumen_{datetime.now().timestamp()}"
            )

            # --- 3) Métricas (UNA sola vez) ---
            total_sub_archivos = len(df_resumen)
            total_archivos_fuente = df_resumen['Archivo_Fuente'].nunique()
            total_registros = df_resumen['Registros'].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Sub-archivos", total_sub_archivos)
            c2.metric("Archivos Fuente", total_archivos_fuente)
            c3.metric("Total Registros", total_registros)

            st.markdown("---")

            # --- 4) Agrupar por archivo origen para mostrar expanders ---
            grouped = {}
            for _, row in df_resumen.iterrows():
                archivo = row['Archivo_Fuente']
                tiempo = row['Tiempo_s']
                grouped.setdefault(archivo, {}).setdefault(tiempo, []).append(row.to_dict())

            # Mostrar 1 expander por archivo origen
            for archivo_origen, tiempos_dict in grouped.items():
                num_tiempos = len(tiempos_dict)
                with st.expander(f"📁 {archivo_origen} - {num_tiempos} tiempos", expanded=False):
                    # Generar ZIP con todos los sub-archivos de este origen (para descargar de una)
                    # Crearlo aquí en memoria y mostrar botón
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for tiempo, items in tiempos_dict.items():
                            for it in items:
                                clave = it['Clave']
                                df_sub = st.session_state.sub_archivos_generados[clave]['datos']
                                csv_bytes = df_sub.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')
                                zf.writestr(it['Nombre_Archivo'], csv_bytes)
                    zip_bytes = buffer.getvalue()
                    st.download_button(
                        label=f"📦 Descargar TODOS los sub-archivos ({archivo_origen}) .zip",
                        data=zip_bytes,
                        file_name=f"{archivo_origen}_subarchivos_{datetime.now().strftime('%Y%m%d')}.zip",
                        mime="application/zip",
                        key=f"zip_{archivo_origen}_{datetime.now().timestamp()}"
                    )

                    st.markdown("")

                    # Para cada tiempo, listar sub-archivos (X) y dar botón CSV por cada uno
                    for tiempo, items in sorted(tiempos_dict.items()):
                        st.markdown(f"#### ⏱️ {tiempo}")
                        for it in sorted(items, key=lambda r: (r['Posicion_X'] if pd.notna(r['Posicion_X']) else 1e9)):
                            clave = it['Clave']
                            nombre = it['Nombre_Archivo']
                            registros = it['Registros']
                            pos_x = it['Posicion_X']

                            col_a, col_b, col_c = st.columns([4, 1, 2])
                            col_a.markdown(f"**X{pos_x}** — `{nombre}`")
                            col_b.markdown(f"Registros: {registros}")

                            # generar bytes CSV para el botón
                            df_sub = st.session_state.sub_archivos_generados[clave]['datos']
                            csv_bytes = df_sub.to_csv(sep=';', index=False, decimal=',').encode('utf-8-sig')

                            # key única por descarga (clave ya debería ser única)
                            dl_key = f"dl_{clave}_{datetime.now().timestamp()}"
                            col_c.download_button(
                                label="📥 Descargar CSV",
                                data=csv_bytes,
                                file_name=nombre,
                                mime="text/csv",
                                key=dl_key,
                                use_container_width=True
                            )

                        st.markdown("---")
        else:
            st.info("No hay sub-archivos generados aún. Subí y procesá archivos en Paso 2.")

        # Paso 4: Sección de Gráficos
        st.markdown("## 📈 Paso 4: Sección de Gráficos")

        if st.session_state.datos_procesados:
            # Tomar el primer DataFrame procesado para obtener n_sensores
            primer_df = next(iter(st.session_state.datos_procesados.values()))
            n_sensores_detectados = primer_df.attrs.get("n_sensores", 12)

            posiciones_sensores = calcular_posiciones_sensores(
                st.session_state.configuracion_inicial['distancia_toma_12'],
                st.session_state.configuracion_inicial['distancia_entre_tomas'],
                n_sensores_detectados,
                st.session_state.configuracion_inicial['orden']
            )

            # Mostrar tabla de posiciones
            with st.expander("Ver posiciones calculadas de sensores"):
                pos_df = pd.DataFrame([
                    {
                        'Sensor': sensor,
                        'Posición Y [mm]': pos['y'],
                        'Sensor Físico': pos['sensor_fisico']
                    }
                    for sensor, pos in posiciones_sensores.items()
                ])
                st.dataframe(pos_df, use_container_width=True)
        else:
            st.warning("⚠️ No hay datos procesados aún. Suba archivos en el Paso 2.")


        # Contenedor para los filtros de visualización
        with st.container(border=True):
            st.markdown("#### 🔍 Filtros de Visualización")
            
            sub_archivos = st.session_state.sub_archivos_generados.values()
            archivos_opciones = sorted(list(set(sa['archivo_fuente'] for sa in sub_archivos)))
            x_opciones = sorted(list(set(sa['x_traverser'] for sa in sub_archivos)))
            tiempo_opciones = sorted(list(set(sa['tiempo'] for sa in sub_archivos)))

            col1, col2, col3 = st.columns(3)
            archivos_seleccionados = col1.multiselect(
                "Filtrar por Archivo Origen:",
                options=archivos_opciones,
                default=archivos_opciones,
                key="filtro_archivos_origen"
            )
            x_seleccionados = col2.multiselect(
                "Filtrar por Posición X:",
                options=x_opciones,
                default=x_opciones,
                key="filtro_posicion_x"
            )
            tiempos_seleccionados = col3.multiselect(
                "Filtrar por Tiempo (s):",
                options=tiempo_opciones,
                default=tiempo_opciones,
                key="filtro_tiempo_s"
            )

        # Filtrar sub-archivos según filtros seleccionados
        sub_archivos_filtrados = {
            clave: sub_archivo for clave, sub_archivo in st.session_state.sub_archivos_generados.items()
            if sub_archivo['archivo_fuente'] in archivos_seleccionados
            and sub_archivo['x_traverser'] in x_seleccionados
            and sub_archivo['tiempo'] in tiempos_seleccionados
        }

        # Selección de sub-archivos para gráfico concatenado
        st.markdown("### 🎯 Selección de Sub-archivos para Gráfico Concatenado")

        if not sub_archivos_filtrados:
            st.warning("No hay datos que coincidan con los filtros seleccionados.")
        else:
            sub_archivos_seleccionados = {}

            # 🎨 Generar color único aleatorio para cada sub-archivo
                # Inicializar colores persistentes en la sesión
            if "colores_por_subarchivo" not in st.session_state:
                st.session_state.colores_por_subarchivo = {}

            # Asignar color solo a los sub-archivos que no tengan uno
            for clave in sub_archivos_filtrados.keys():
                if clave not in st.session_state.colores_por_subarchivo:
                    st.session_state.colores_por_subarchivo[clave] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

            colores_por_subarchivo = st.session_state.colores_por_subarchivo

            # Mostrar lista de selección con colores
            for i, (clave, sub_archivo) in enumerate(sorted(sub_archivos_filtrados.items())):
                col1, col2 = st.columns([3, 1])
                label = f"{sub_archivo['archivo_fuente']} - T{sub_archivo['tiempo']}s - X{sub_archivo['x_traverser']} - {len(sub_archivo['datos'])} registros"
                
                if col1.checkbox(label, key=f"incluir_{clave}_{i}"):
                    sub_archivos_seleccionados[clave] = sub_archivo
                
                with col2:
                    color_sub = colores_por_subarchivo[clave]
                    st.markdown(
                        f'<div style="background: {color_sub}; height: 20px; width: 60px; border-radius: 3px; margin-top: 8px;"></div>',
                        unsafe_allow_html=True
                    )

            # Generar gráfico concatenado si hay selecciones
            if sub_archivos_seleccionados:
                st.markdown("### 📊 Gráfico Concatenado Vertical Modo Betz")

                fig = go.Figure()
                for clave, sub_archivo in sub_archivos_seleccionados.items():
                    color = colores_por_subarchivo[clave]
                    z_datos, presion_datos = extraer_datos_para_grafico(sub_archivo, st.session_state.configuracion_inicial)
                    if z_datos and presion_datos:
                        fig.add_trace(go.Scatter(
                            x=presion_datos,
                            y=z_datos,
                            mode='lines',
                            name=clave,
                            line=dict(color=color, width=2),
                            fill='tozerox',
                            opacity=0.7
                        ))

                fig.update_layout(
                    title="Perfil de Presión Concatenado",
                    xaxis_title="Presión Total [Pa]",
                    yaxis_title="Altura Z [mm]",
                    plot_bgcolor='rgba(0,0,0,0)',   # transparente en área del gráfico
                    paper_bgcolor='rgba(0,0,0,0)',  # transparente en todo el lienzo
                    font=dict(color='white'),       # texto en blanco para que se lea bien
                    height=900,
                    width=1600
                )
                st.plotly_chart(fig, use_container_width=False)

                total_puntos = len(sub_archivos_seleccionados) * 12
                st.success(f"✅ Gráfico vertical generado con {total_puntos} puntos de datos concatenados")

                # Exportaciones
                st.markdown("### 📤 Exportación")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    html_string = fig.to_html()
                    st.download_button(
                        label="📊 Descargar Gráfico (HTML)",
                        data=html_string,
                        file_name=f"grafico_betz_vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )

                with col2:
                    datos_exportar = []
                    for clave, sub_archivo in sub_archivos_seleccionados.items():
                        datos_sub = sub_archivo['datos'].copy()
                        datos_sub['Sub_archivo'] = clave
                        datos_exportar.append(datos_sub)
                    
                    df_exportar = pd.concat(datos_exportar, ignore_index=True)
                    csv_data = df_exportar.to_csv(
                        index=False, sep=';', encoding='utf-8-sig', decimal=','
                    )
                    st.download_button(
                        label="📋 Descargar Datos (CSV)",
                        data=csv_data,
                        file_name=f"datos_betz_vertical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col3:
                    st.metric("Total de puntos", total_puntos)
            else:
                st.error("❌ No se pudo generar el gráfico. Verifique los datos seleccionados.")

        
            # NUEVA SECCIÓN: RESTA DE ÁREAS
            # NUEVA SECCIÓN: RESTA DE ÁREAS
            if st.session_state.sub_archivos_generados:
                st.markdown("---")
                st.markdown("## ➖ Análisis de Diferencias de Áreas")
                st.markdown("Selecciona dos sub-archivos para calcular la diferencia de áreas entre sus perfiles de presión")
                
                # Crear lista de opciones para los selectores
                opciones_subarchivos = sorted(list(st.session_state.sub_archivos_generados.keys()))
                
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    archivo_a = st.selectbox(
                        "📊 Archivo A (minuendo):",
                        opciones_subarchivos,
                        key="archivo_a_resta",
                        help="Seleccione el primer sub-archivo (del cual se restará el segundo)"
                    )
                
                with col2:
                    st.markdown("<div style='text-align: center; font-size: 2rem; margin-top: 2rem;'>➖</div>", unsafe_allow_html=True)
                
                with col3:
                    archivo_b = st.selectbox(
                        "📊 Archivo B (sustraendo):",
                        opciones_subarchivos,
                        index=1 if len(opciones_subarchivos) > 1 else 0,
                        key="archivo_b_resta",
                        help="Seleccione el segundo sub-archivo (que será restado del primero)"
                    )
                
                # --- INICIO DE LA ESTRUCTURA CORREGIDA ---

                # PARTE 1: Botón para CALCULAR. Su única misión es generar el gráfico y ponerlo en una "bandeja" temporal.
                if st.button("🔄 Calcular Diferencia de Áreas", type="primary", use_container_width=True):
                    if archivo_a and archivo_b and archivo_a != archivo_b:
                        with st.spinner("Calculando diferencia de áreas..."):
                            fig_diferencia, diferencia_area = crear_grafico_diferencia_areas(
                                st.session_state.sub_archivos_generados[archivo_a],
                                st.session_state.sub_archivos_generados[archivo_b],
                                st.session_state.configuracion_inicial
                            )
                            if fig_diferencia:
                                # Guardamos TODO lo necesario en la sesión para mostrarlo después del reinicio de la página.
                                st.session_state.figura_diferencia_temporal = {
                                    "fig": fig_diferencia,
                                    "nombre": f"Dif. 2D: {archivo_a.split('_')[0]} vs {archivo_b.split('_')[0]}",
                                    "area_diferencia": diferencia_area,
                                    "archivo_a": archivo_a,
                                    "archivo_b": archivo_b
                                }
                            else:
                                st.error("❌ No se pudo calcular la diferencia.")
                                if 'figura_diferencia_temporal' in st.session_state:
                                    del st.session_state.figura_diferencia_temporal
                    else:
                        st.warning("⚠️ Seleccione dos sub-archivos diferentes para calcular la diferencia.")

                # PARTE 2: Este bloque está AFUERA del anterior. Revisa si hay algo en la "bandeja" temporal.
                # Si hay algo, lo muestra junto con TODOS sus botones (Guardar, métricas, descarga).
                if 'figura_diferencia_temporal' in st.session_state:
                    # Recuperamos los datos de la sesión que guardamos en la PARTE 1
                    temp_data = st.session_state.figura_diferencia_temporal
                    fig_diferencia = temp_data["fig"]
                    nombre_guardado = temp_data["nombre"]
                    diferencia_area = temp_data["area_diferencia"]
                    archivo_a_calc = temp_data["archivo_a"]
                    archivo_b_calc = temp_data["archivo_b"]
                    
                    # 1. Mostramos el gráfico
                    st.plotly_chart(fig_diferencia, use_container_width=False)
                    
                    # 2. Mostramos el botón de "Guardar". Ahora sí funcionará.
                    if st.button("💾 Guardar Diferencia para Visualizar", key="save_diff_2d_for_viz_final"):
                        if 'diferencias_guardadas' not in st.session_state:
                            st.session_state.diferencias_guardadas = {}
                        st.session_state.diferencias_guardadas[nombre_guardado] = fig_diferencia
                        st.success(f"✅ Gráfico '{nombre_guardado}' guardado permanentemente.")
                        # Borramos la figura temporal después de guardarla para limpiar la "bandeja"
                        del st.session_state.figura_diferencia_temporal
                        st.rerun()

                    # 3. Mostramos las métricas y el resto de tu código (sin cambios)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    z_a, p_a = extraer_datos_para_grafico(st.session_state.sub_archivos_generados[archivo_a_calc], st.session_state.configuracion_inicial)
                    z_b, p_b = extraer_datos_para_grafico(st.session_state.sub_archivos_generados[archivo_b_calc], st.session_state.configuracion_inicial)
                    area_a = calcular_area_bajo_curva(z_a, p_a)
                    area_b = calcular_area_bajo_curva(z_b, p_b)
                    
                    with col_m1:
                        st.metric(f"Área {st.session_state.sub_archivos_generados[archivo_a_calc]['archivo_fuente']}", f"{area_a:.2f} Pa·mm")
                    with col_m2:
                        st.metric(f"Área {st.session_state.sub_archivos_generados[archivo_b_calc]['archivo_fuente']}", f"{area_b:.2f} Pa·mm")
                    with col_m3:
                        st.metric("Diferencia de Áreas", f"{diferencia_area:.2f} Pa·mm", delta=f"{diferencia_area:.2f}", delta_color="normal" if diferencia_area >= 0 else "inverse")
                    
                    if diferencia_area > 0:
                        st.success(f"✅ El área de **{st.session_state.sub_archivos_generados[archivo_a_calc]['archivo_fuente']}** es **{diferencia_area:.2f} Pa·mm** mayor que la de **{st.session_state.sub_archivos_generados[archivo_b_calc]['archivo_fuente']}**")
                    elif diferencia_area < 0:
                        st.info(f"ℹ️ El área de **{st.session_state.sub_archivos_generados[archivo_b_calc]['archivo_fuente']}** es **{abs(diferencia_area):.2f} Pa·mm** mayor que la de **{st.session_state.sub_archivos_generados[archivo_a_calc]['archivo_fuente']}**")
                    else:
                        st.info("ℹ️ Las áreas son prácticamente iguales")
                    
                    html_diferencia = fig_diferencia.to_html()
                    st.download_button(
                        label="📊 Descargar Gráfico de Diferencia (HTML)",
                        data=html_diferencia,
                        file_name=f"diferencia_areas_{archivo_a_calc}_vs_{archivo_b_calc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )

elif st.session_state.seccion_actual == 'betz_3d':
    st.markdown("# 🌪️ BETZ 3D - Análisis Tridimensional")
    st.markdown("Análisis 3D con superficie interactiva de presiones")
    
    # Paso 1: Configuración inicial
    st.markdown("## ⚙️ Paso 1: Configuración Inicial")

    # Reorganizar: datos a la izquierda, imagen más pequeña a la derecha
    col_datos, col_imagen = st.columns([2, 1])

    with col_datos:
        st.markdown("### 📍 Configuración de Sensores y Geometría")
        
        # Orden de sensores
        orden_sensores = st.selectbox(
            "Orden de lectura de sensores:",
            ["asc", "des"],
            format_func=lambda x: "Ascendente (sensor 1 más abajo al 12 más arriba)" if x == "asc" else "Descendente (sensor 12 más abajo y sensor 1 más arriba)",
            help="Define cómo se leen los datos de los sensores en relación a su posición física",
            key="orden_3d"
        )
        
        # Pregunta sobre el sensor de referencia
        st.info("🔍 **Pregunta:** ¿Qué sensor corresponde a la toma número 12 (la que se encuentra cerca del piso)?")
        sensor_referencia = st.selectbox(
            "Sensor de referencia (toma 12):",
            [f"Sensor {i}" for i in range(1, 13)],
            index=11,  # Por defecto Sensor 12
            help="Seleccione el sensor que corresponde a la toma física número 12",
            key="sensor_ref_3d"
        )
        
        distancia_toma_12 = st.number_input(
            "Distancia de la toma 12 a la posición X=0, Y=0 (coordenadas del traverser) [mm]:",
            value=-120.0,
            step=1.0,
            format="%.1f",
            help="Distancia en mm desde el punto de referencia del traverser",
            key="dist_toma_3d"
        )
        
        distancia_entre_tomas = st.number_input(
            "Distancia entre tomas [mm]:",
            value=10.91,
            step=0.01,
            format="%.2f",
            help="Distancia física entre tomas consecutivas según el plano técnico",
            key="dist_entre_3d"
        )
        
        # Guardar configuración
        if st.button("💾 Guardar Configuración 3D", type="primary", key="save_3d"):
            st.session_state.configuracion_3d = {
                'orden': orden_sensores,
                'sensor_referencia': sensor_referencia,
                'distancia_toma_12': distancia_toma_12,
                'distancia_entre_tomas': distancia_entre_tomas
            }
            st.success("✅ Configuración 3D guardada correctamente")
            st.rerun()

    with col_imagen:
        st.markdown("### 📐 Diagrama de Referencia")
        st.markdown("""
        <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2rem; text-align: center; color: #64748b;">
            <h4>📐 Diagrama de Referencia</h4>
            <p>Aquí iría el diagrama técnico de sensores</p>
            <p><small>Subir imagen del plano técnico</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar configuración actual
    if st.session_state.get('configuracion_3d'): # <-- MODIFICA ESTA LÍNEA
        st.markdown("## ⚙️ Configuración 3D Actual")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Orden sensores", st.session_state.configuracion_3d['orden'].upper())
        with col2:
            st.metric("Distancia toma 12", f"{st.session_state.configuracion_3d['distancia_toma_12']:.1f} mm")
        with col3:
            st.metric("Distancia entre tomas", f"{st.session_state.configuracion_3d['distancia_entre_tomas']:.2f} mm")
        
        # Paso 2: Carga de archivos 3D
        st.markdown("## 📁 Paso 2: Carga de Archivos 3D")

        # Almacenar múltiples archivos 3D
        if 'archivos_3d_cargados' not in st.session_state:
            st.session_state.archivos_3d_cargados = {}

        uploaded_files_3d = []
        uploaded_files_3d = st.file_uploader(
        "Seleccione uno o más archivos CSV:",
        type=['csv'],
        accept_multiple_files=True,
        key="uploader_betz3d"  # clave única para BETZ 3D
    )
        if uploaded_files_3d:
            for uploaded_file_3d in uploaded_files_3d:
                nombre_archivo = uploaded_file_3d.name.replace('.csv', '').replace('incertidumbre_', '')
                
                if nombre_archivo not in st.session_state.archivos_3d_cargados:
                    with st.spinner(f"Procesando {nombre_archivo}..."):
                        # Procesar archivo 3D
                        datos_3d = procesar_promedios(uploaded_file_3d, st.session_state.configuracion_3d['orden'])
                        
                        if datos_3d is not None:
                            st.session_state.archivos_3d_cargados[nombre_archivo] = datos_3d
                            
                            # Crear sub-archivos 3D por tiempo
                            sub_archivos_3d = crear_sub_archivos_3d_por_tiempo_y_posicion(datos_3d, nombre_archivo)
                            st.session_state.sub_archivos_3d_generados.update(sub_archivos_3d)
                            
                            st.success(f"✅ {nombre_archivo} procesado: {len(datos_3d)} registros")

        # Mostrar archivos cargados
        if st.session_state.archivos_3d_cargados:
            st.markdown("### 📋 Archivos 3D Cargados")
            
            for nombre, datos in st.session_state.archivos_3d_cargados.items():
                with st.expander(f"📁 {nombre}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        tiempos_unicos = datos['Tiempo (s)'].dropna().unique()
                        st.metric("Tiempos", len(tiempos_unicos))
                        st.write(f"T: {sorted(tiempos_unicos)}")
                    with col2:
                        posiciones_x = datos['X_coord'].dropna().unique()
                        st.metric("Posiciones X", len(posiciones_x))
                        st.write(f"X: {sorted(posiciones_x)}")
                    with col3:
                        posiciones_y = datos['Y_coord'].dropna().unique()
                        st.metric("Posiciones Y", len(posiciones_y))
                        st.write(f"Y: {sorted(posiciones_y)}")
                    with col4:
                        st.metric("Total registros", len(datos))

        # NUEVO: Controles de Visualización (mover antes de la visualización individual)
        if st.session_state.archivos_3d_cargados:
            st.markdown("### 🎛️ Controles de Visualización")
            
            col1, col2 = st.columns(2)
            with col1:
                mostrar_cuerpo = st.checkbox(
                    "🏗️ Mostrar cuerpo sólido debajo",
                    value=True,
                    help="Agrega una base sólida debajo de la superficie de presiones",
                    key="mostrar_cuerpo_global"
                )
            
            with col2:
                aspect_ratio = st.selectbox(
                    "📐 Proporción de aspecto:",
                    ["auto", "equal", "manual"],
                    format_func=lambda x: {
                        "auto": "Automático",
                        "equal": "Igual (cubo)",
                        "manual": "Manual (2:2:1)"
                    }[x],
                    help="Controla las proporciones del gráfico 3D",
                    key="aspect_ratio_global"
                )

        # Paso 3: Visualización Individual de Archivos 3D
        st.markdown("## 🌪️ Paso 3: Visualización Individual de Archivos 3D")
        st.markdown("Selecciona un archivo para ver su superficie 3D individual")
        
        # 🔘 Checkbox para activar/desactivar puntos medidos
        mostrar_puntos_3d = st.checkbox("Mostrar puntos medidos en la superficie", value=True, key="mostrar_puntos_3d")

        if st.session_state.archivos_3d_cargados:
            # Crear interfaz de selección tipo browser
            st.markdown("### 📁 Archivos 3D Disponibles")
            
            # Mostrar archivos como botones seleccionables
            archivos_disponibles = list(st.session_state.archivos_3d_cargados.keys())
            
            # Crear columnas para mostrar archivos
            cols_per_row = 3
            for i in range(0, len(archivos_disponibles), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(archivos_disponibles):
                        nombre_archivo = archivos_disponibles[i + j]
                        datos_archivo = st.session_state.archivos_3d_cargados[nombre_archivo]
                        
                        with col:
                            # Crear card para cada archivo
                            st.markdown(f"""
                            <div style="
                                background: white;
                                border: 2px solid #e5e7eb;
                                border-radius: 12px;
                                padding: 1rem;
                                margin: 0.5rem 0;
                                text-align: center;
                                transition: all 0.3s ease;
                                cursor: pointer;
                            ">
                                <h4 style="color: #08596C; margin-bottom: 0.5rem;">📊 {nombre_archivo}</h4>
                                <p style="color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;">
                                    {len(datos_archivo)} registros
                                </p>
                                <p style="color: #6b7280; font-size: 0.8rem;">
                                    Tiempos: {len(datos_archivo['Tiempo (s)'].dropna().unique())}<br>
                                    Pos. X: {len(datos_archivo['X_coord'].dropna().unique())}<br>
                                    Pos. Y: {len(datos_archivo['Y_coord'].dropna().unique())}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Botón para visualizar este archivo
                            # Nuevo bloque
                            # Nuevo Bloque Definitivo
                            # NUEVO BLOQUE DEL BOTÓN. Reemplaza el anterior con este.
                            if st.button(f"🏔️ Ver Superficie Completa", key=f"ver_mesh3d_{nombre_archivo}", use_container_width=True):
                                with st.spinner(f"Construyendo superficie completa para {nombre_archivo}..."):
                                    # Llamada a la NUEVA función de graficación con 300 puntos
                                    fig_individual = crear_superficie_delaunay_3d(
                                        datos_archivo,
                                        st.session_state.configuracion_3d,
                                        nombre_archivo,
                                        mostrar_puntos=mostrar_puntos_3d  # ← Aquí
                                    )
                                    
                                    if fig_individual:
                                        st.plotly_chart(fig_individual, use_container_width=False)
                                        
                                        # Información del archivo
                                        st.success(f"✅ Superficie de malla 3D generada para: **{nombre_archivo}** usando {len(fig_individual.data[0].x)} vértices.")
                                        
                                        # Botón de descarga
                                        html_individual = fig_individual.to_html()
                                        st.download_button(
                                            label=f"📊 Descargar Malla 3D (HTML)",
                                            data=html_individual,
                                            file_name=f"mesh_3d_{nombre_archivo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                            mime="text/html",
                                            key=f"download_mesh3d_{nombre_archivo}"
                                        )
                                    else:
                                        st.error(f"❌ No se pudo generar la superficie de malla 3D para {nombre_archivo}")   
        # --- NUEVO PASO 4: DIFERENCIA ENTRE SUPERFICIES ---
        # --- PASO 4: DIFERENCIA ENTRE SUPERFICIES (Delaunay) ---
        st.markdown("## ➖ Paso 4: Diferencia entre Superficies (Delaunay)")
        if len(st.session_state.get('archivos_3d_cargados', {})) >= 2:
            st.info("Calcula la diferencia (A - B) entre dos superficies usando triangulación de Delaunay.")

            archivos_disponibles_diff = list(st.session_state.get('archivos_3d_cargados', {}).keys())

            col1_diff, col2_diff = st.columns(2)
            with col1_diff:
                archivo_a = st.selectbox(
                    "Superficie A (Minuendo):",
                    archivos_disponibles_diff,
                    key="diff_3d_a"
                )
            with col2_diff:
                archivo_b = st.selectbox(
                    "Superficie B (Sustraendo):",
                    archivos_disponibles_diff,
                    index=1 if len(archivos_disponibles_diff) > 1 else 0,
                    key="diff_3d_b"
                )
            
            # 🔘 Checkbox para activar/desactivar puntos en diferencia
            mostrar_puntos_diff = st.checkbox("Mostrar puntos medidos en la diferencia", value=True, key="mostrar_puntos_diff")
            # --- INICIO DE LA ESTRUCTURA CORREGIDA ---

            # PARTE 1: El botón "Calcular" solo genera el gráfico y lo guarda en una "bandeja" temporal.
            if st.button("Calcular Diferencia de Superficies", use_container_width=True, type="primary"):
                if archivo_a == archivo_b:
                    st.error("Por favor, selecciona dos archivos diferentes para comparar.")
                else:
                    with st.spinner(f"Calculando diferencia entre '{archivo_a}' y '{archivo_b}'..."):
                        datos_a = st.session_state.archivos_3d_cargados[archivo_a]
                        datos_b = st.session_state.archivos_3d_cargados[archivo_b]
                        
                        fig_diferencia_3d = crear_superficie_diferencia_delaunay_3d(
                            datos_a,
                            datos_b,
                            archivo_a,
                            archivo_b,
                            st.session_state.configuracion_3d,
                            mostrar_puntos=mostrar_puntos_diff  # ← Aquí
                        )

                        if fig_diferencia_3d:
                            # Guardamos la figura y su nombre en la sesión para que sobrevivan al reinicio de la página.
                            st.session_state.figura_diferencia_temporal_3d = {
                                "fig": fig_diferencia_3d,
                                "nombre": f"Dif. 3D: {archivo_a} vs {archivo_b}"
                            }
                        else:
                            if 'figura_diferencia_temporal_3d' in st.session_state:
                                del st.session_state.figura_diferencia_temporal_3d
            
            # PARTE 2: Este bloque (fuera del anterior) revisa si hay algo en la "bandeja" temporal y lo muestra.
            if 'figura_diferencia_temporal_3d' in st.session_state:
                temp_data_3d = st.session_state.figura_diferencia_temporal_3d
                fig_diferencia_3d = temp_data_3d["fig"]
                nombre_guardado_3d = temp_data_3d["nombre"]

                # 1. Mostramos el gráfico
                st.plotly_chart(fig_diferencia_3d, use_container_width=False)
                
                # 2. Mostramos el botón "Guardar". Ahora sí funcionará.
                if st.button("💾 Guardar Diferencia para Visualizar", key="save_diff_3d_for_viz_final"):
                    if 'diferencias_guardadas' not in st.session_state:
                        st.session_state.diferencias_guardadas = {}
                    
                    st.session_state.diferencias_guardadas[nombre_guardado_3d] = fig_diferencia_3d
                    st.success(f"✅ Gráfico '{nombre_guardado_3d}' guardado permanentemente.")
                    
                    # Borramos la figura temporal después de guardarla para limpiar la "bandeja"
                    del st.session_state.figura_diferencia_temporal_3d
                    st.rerun()
            # --- FIN DE LA ESTRUCTURA CORREGIDA ---

        else:
            st.info("Carga al menos dos archivos 3D para poder calcular una diferencia.")

    else:
        st.info("⚙️ Complete la configuración 3D para continuar")

elif st.session_state.seccion_actual == 'visualizacion':
    st.markdown("# 🖥️ Visualización de Resultados")
    st.markdown("Agrega y compara gráficos 2D y 3D generados en BETZ.")

    # Inicializar contenedores de guardado si no existen
    if "graficos_guardados" not in st.session_state:
        st.session_state.graficos_guardados = []

    if "diferencias_guardadas" not in st.session_state:
        st.session_state.diferencias_guardadas = {}

    # Variables de estado seguras (evitan KeyError)
    sub_archivos_2d = st.session_state.get("sub_archivos_generados", {})
    sub_archivos_3d = st.session_state.get("sub_archivos_3d_generados", {})
    config_2d = st.session_state.get("configuracion_inicial", {})
    config_3d = st.session_state.get("configuracion_3d", {})
    diferencias_guardadas = st.session_state.get("diferencias_guardadas", {})

    # --- Selector único que une 2D, 3D y diferencias guardadas ---
    opciones = list(sub_archivos_2d.keys()) + list(sub_archivos_3d.keys()) + list(diferencias_guardadas.keys())
    if st.session_state.get("configuracion_inicial"):
        posiciones_sensores = calcular_posiciones_sensores(
            st.session_state.configuracion_inicial['distancia_toma_12'],
            st.session_state.configuracion_inicial['distancia_entre_tomas'],
            st.session_state.configuracion_inicial['orden']
        )
    else:
        posiciones_sensores = {}
    if opciones:
        seleccionado = st.selectbox("Selecciona un resultado para agregar", opciones)

        if st.button("➕ Agregar a Visualización"):
            fig = None

            # Caso: sub-archivo 2D
            if seleccionado in sub_archivos_2d:
                # crear_grafico_betz_concatenado espera un dict de sub-archivos seleccionados
                fig = crear_grafico_betz_concatenado(
                    {seleccionado: sub_archivos_2d[seleccionado]},
                    posiciones_sensores,
                    st.session_state.configuracion_inicial
                )

            # Caso: sub-archivo 3D
            elif seleccionado in sub_archivos_3d:
                fig_data = sub_archivos_3d[seleccionado]
                fig = crear_superficie_delaunay_3d(
                    fig_data.get('datos', pd.DataFrame()),
                    config_3d,
                    fig_data.get('archivo_fuente', seleccionado)
                )

            # Caso: gráfico de diferencia ya guardado (2D o 3D)
            elif seleccionado in diferencias_guardadas:
                fig = diferencias_guardadas[seleccionado]

            # Si obtuvimos una figura, la guardamos en la lista de gráficos agregados
            if fig:
                # Guardamos una tupla (titulo, figura) para mostrar título más tarde
                st.session_state.graficos_guardados.append((seleccionado, fig))
                st.success(f"✅ '{seleccionado}' agregado a la visualización")
                st.rerun()
    else:
        st.info("No hay resultados 2D o 3D disponibles para visualizar. Procesa/guarda alguno en BETZ 2D o BETZ 3D.")

    # --- SECCIÓN PARA MOSTRAR Y ELIMINAR LOS GRÁFICOS AGREGADOS ---
    if st.session_state.graficos_guardados:
        st.markdown("---")
        st.markdown("### Gráficos Agregados")

        if st.button("🗑️ Limpiar todos los gráficos"):
            st.session_state.graficos_guardados = []
            st.rerun()

        cols = st.columns(2)

        # Iteramos sobre una copia de la lista para poder modificar la original
        for i, (titulo, fig) in enumerate(list(st.session_state.graficos_guardados)):
            with cols[i % 2]:
                st.markdown(f"#### {titulo}")
                st.plotly_chart(fig, use_container_width=True)

                # Botón para eliminar ese gráfico concreto
                # Key única que combina índice + título (evita colisiones)
                if st.button("❌ Eliminar Gráfico", key=f"eliminar_{i}_{titulo}"):
                    # eliminar por índice
                    st.session_state.graficos_guardados.pop(i)
                    st.success(f"Gráfico '{titulo}' eliminado")
                    st.rerun()

    # (Opcional) zona para descargas globales o métricas adicionales
    st.markdown("---")
    st.markdown("### 🔽 Descargas / Resumen rápido")
    if st.session_state.graficos_guardados:
        st.write(f"Gráficos en el lienzo: {len(st.session_state.graficos_guardados)}")
    else:
        st.write("No hay gráficos en el lienzo actualmente.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Laboratorio de Aerodinámica y Fluidos - UTN HAEDO</strong></p>
    <p>Sistema de Análisis de Datos Aerodinámicos • Versión 1.37 - </p>
    <p><small>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small></p>
</div>
""", unsafe_allow_html=True)



