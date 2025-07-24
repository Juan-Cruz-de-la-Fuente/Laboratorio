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

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio de Aerodinámica y Fluidos - UTN HAEDO",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def extraer_tiempo_y_coordenadas(nombre_archivo):
    """Extraer tiempo y coordenadas X, Y del nombre del archivo"""
    tiempo = None
    x_coord = None
    y_coord = None
    
    # Extraer tiempo después de "T" (ej: T10s, T20s)
    tiempo_match = re.search(r"[Tt](\d+)s", nombre_archivo)
    if tiempo_match:
        tiempo = int(tiempo_match.group(1))
    
    # Extraer coordenada X (ej: X-0, X0, X=0)
    x_match = re.search(r"[Xx][-=]?(\d+)", nombre_archivo)
    if x_match:
        x_coord = int(x_match.group(1))
    
    # Extraer coordenada Y (ej: Y-0, Y0, Y=0)  
    y_match = re.search(r"[Yy][-=]?(\d+)", nombre_archivo)
    if y_match:
        y_coord = int(y_match.group(1))
    
    return tiempo, x_coord, y_coord

def extraer_nombre_base_archivo(nombre_archivo):
    """Extraer nombre base del archivo (sin extensión y sin 'incertidumbre_')"""
    nombre_base = nombre_archivo.replace('.csv', '').replace('incertidumbre_', '').replace('_', ' ')
    # Capitalizar primera letra de cada palabra
    return ' '.join(word.capitalize() for word in nombre_base.split())

def procesar_promedios(archivo_csv, orden="asc"):
    """Procesar archivo de incertidumbre siguiendo exactamente el código de referencia"""
    try:
        # Leer el archivo CSV exactamente como en el código de referencia
        df_raw = pd.read_csv(archivo_csv, sep=";", header=None)
        
        # Buscar la palabra "importante" para determinar dónde terminar
        index_final = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("importante", case=False).any(), axis=1)].index
        if not index_final.empty:
            df_raw = df_raw.iloc[:index_final[0]]

        resultados = []

        # Procesar bloques de 10 filas exactamente como en el código de referencia
        for i in range(0, df_raw.shape[0], 10):
            bloque = df_raw.iloc[i:i+10]
            if bloque.empty or len(bloque) < 3:
                continue

            archivo = bloque.iloc[0, 0]
            sensores = bloque.iloc[0, 1:].tolist()
            valores = bloque.iloc[2, 1:].tolist()
            muestras = bloque.iloc[1, 1]

            # Invertir orden si es necesario
            if orden == "des":
                valores = valores[::-1]

            fila = {
                "Archivo": archivo,
                "Numero de muestras": muestras,
            }
            for sensor, valor in zip(sensores, valores):
                fila[sensor] = valor

            resultados.append(fila)

        df_resultado = pd.DataFrame(resultados)

        # Extraer tiempo y coordenadas desde nombre de archivo
        coordenadas_tiempo = df_resultado["Archivo"].apply(extraer_tiempo_y_coordenadas)
        
        df_resultado["Tiempo (s)"] = [coord[0] for coord in coordenadas_tiempo]
        df_resultado["X_coord"] = [coord[1] for coord in coordenadas_tiempo]
        df_resultado["Y_coord"] = [coord[2] for coord in coordenadas_tiempo]

        return df_resultado
    
    except Exception as e:
        st.error(f"Error al procesar archivo: {str(e)}")
        return None

def crear_archivos_individuales_por_tiempo_y_posicion(df_resultado, nombre_archivo_fuente):
    """Crear archivos individuales SOLO por tiempo (consolidando todas las posiciones)"""
    sub_archivos = {}
    
    # Extraer nombre base del archivo fuente
    nombre_base = extraer_nombre_base_archivo(nombre_archivo_fuente)
    
    # Obtener tiempos únicos
    tiempos_unicos = df_resultado["Tiempo (s)"].dropna().unique()
    
    for tiempo in tiempos_unicos:
        # Filtrar datos por tiempo (TODAS las posiciones)
        df_tiempo = df_resultado[df_resultado["Tiempo (s)"] == tiempo].copy()
        
        # Crear clave para el sub-archivo SOLO CON TIEMPO
        clave_sub_archivo = f"{nombre_base}_T{tiempo}s"
        
        sub_archivos[clave_sub_archivo] = {
            'archivo_fuente': nombre_base,
            'tiempo': tiempo,
            'datos': df_tiempo,  # TODOS los datos de este tiempo
            'nombre_archivo': f"{nombre_base}_T{tiempo}s.csv"
        }
    
    return sub_archivos

def calcular_posiciones_sensores(distancia_toma_12, distancia_entre_tomas, orden="asc"):
    """Calcular posiciones geométricas de los sensores según el orden especificado"""
    posiciones = {}
    
    if orden == "asc":
        # Ascendente: sensor 1 más abajo, sensor 12 más arriba
        for i in range(1, 13):  # Sensores 1 a 12
            y_position = distancia_toma_12 + (i - 1) * distancia_entre_tomas
            posiciones[f"Presion-Sensor {i}"] = {
                'x': 0,
                'y': y_position,
                'sensor_fisico': i
            }
    else:  # "des"
        # Descendente: sensor 12 más abajo, sensor 1 más arriba
        for i in range(1, 13):  # Sensores 1 a 12
            y_position = distancia_toma_12 + (12 - i) * distancia_entre_tomas
            posiciones[f"Presion-Sensor {i}"] = {
                'x': 0,
                'y': y_position,
                'sensor_fisico': i
            }
    
    return posiciones

def crear_grafico_betz_concatenado(sub_archivos_seleccionados, posiciones_sensores, configuracion):
    """Crear gráfico BETZ concatenado vertical (altura vs presión) agrupado por tiempo"""
    
    # Configuración basada en el código original
    posicion_inicial = configuracion['distancia_toma_12']
    distancia_entre_tomas = configuracion['distancia_entre_tomas']
    
    n_tomas = 12
    
    # Paleta de colores por tiempo
    colores_por_tiempo = {
        10: '#08596C',   # Teal
        20: '#E74C3C',   # Rojo
        30: '#F39C12',   # Naranja
        40: '#27AE60',   # Verde
        50: '#8E44AD',   # Púrpura
        60: '#3498DB',   # Azul claro
        70: '#E67E22',   # Naranja oscuro
        80: '#2ECC71',   # Verde claro
        90: '#9B59B6',   # Púrpura claro
        100: '#1ABC9C',  # Turquesa
        110: '#F1C40F',  # Amarillo
        120: '#34495E'   # Gris azulado
    }
    
    # Crear el gráfico
    fig = go.Figure()
    
    # Agrupar sub-archivos por tiempo
    datos_por_tiempo = {}
    for clave, sub_archivo in sub_archivos_seleccionados.items():
        tiempo = sub_archivo['tiempo']
        if tiempo not in datos_por_tiempo:
            datos_por_tiempo[tiempo] = []
        datos_por_tiempo[tiempo].append((clave, sub_archivo))
    
    # Procesar cada tiempo
    for tiempo in sorted(datos_por_tiempo.keys()):
        # Obtener color para este tiempo
        color = colores_por_tiempo.get(tiempo, '#08596C')
        
        # Listas para concatenar todos los datos de este tiempo
        z_tiempo = []
        presion_tiempo = []
        
        # Procesar cada sub-archivo de este tiempo
        for clave, sub_archivo in datos_por_tiempo[tiempo]:
            datos_tiempo = sub_archivo['datos']
            
            # Procesar TODAS las filas (múltiples posiciones) en este tiempo
            for _, fila in datos_tiempo.iterrows():
                y_traverser = fila['Y_coord']  # Obtener Y_coord de cada fila
                
                # Procesar cada toma física
                for toma_fisica in range(1, n_tomas + 1):
                    if configuracion['orden'] == "asc":
                        sensor_columna = f"Presion-Sensor {toma_fisica}"
                    else:
                        sensor_columna = f"Presion-Sensor {13 - toma_fisica}"
                    
                    if sensor_columna in datos_tiempo.columns:
                        # Calcular posición Z total (altura)
                        z_total = y_traverser + (toma_fisica - 1) * distancia_entre_tomas
                        
                        # Obtener presión de esta fila específica
                        presion = fila[sensor_columna]
                        
                        try:
                            if isinstance(presion, str):
                                presion = float(presion.replace(',', '.'))
                            
                            z_tiempo.append(z_total)
                            presion_tiempo.append(presion)
                            
                        except:
                            continue
        
        # Ordenar los puntos de este tiempo por altura (Z)
        if z_tiempo and presion_tiempo:
            datos_ordenados = sorted(zip(z_tiempo, presion_tiempo))
            z_ordenado, presion_ordenada = zip(*datos_ordenados)
            
            # Agregar línea para este tiempo (VERTICAL: X=presión, Y=altura)
            fig.add_trace(go.Scatter(
                x=presion_ordenada,  # Presión en X
                y=z_ordenado,       # Altura en Y
                mode='lines+markers',
                name=f"T{tiempo}s",
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color),
                hovertemplate=f'<b>T{tiempo}s</b><br>' +
                            'Presión: %{x:.3f} Pa<br>' +
                            'Altura: %{y:.1f} mm<br>' +
                            '<extra></extra>'
            ))
            
            # Agregar área sombreada para este tiempo - CORREGIDA
            if len(presion_ordenada) > 1:
                # Crear área desde X=0 hasta la curva
                x_area = [0] + list(presion_ordenada) + [0]
                y_area = [z_ordenado[0]] + list(z_ordenado) + [z_ordenado[-1]]
                
                fig.add_trace(go.Scatter(
                    x=x_area,
                    y=y_area,
                    fill='toself',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Configurar el layout VERTICAL - MÁS COMPACTO
    fig.update_layout(
        title="Perfil de presión concatenado vertical (modo Betz)",
        xaxis_title="Presión total [Pa]",  # X = Presión
        yaxis_title="Altura z [mm]",       # Y = Altura
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,  # Más alto para gráfico vertical
        width=600,   # NUEVO: Ancho más compacto (1/3 menos)
        hovermode='closest',
        font=dict(family="Arial", size=12),
        title_font=dict(size=20, color="#08596C"),
        title_x=0.5,
        xaxis=dict(
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            side='bottom',
            fixedrange=False,
            autorange=False,      # Desactiva autorange
            scaleanchor="y",
            scaleratio=4
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,  # Línea del origen más gruesa
            side='left'
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1,
            x=1.02,
            y=1
        )
    )
    
    return fig

def extraer_datos_para_grafico(sub_archivo, configuracion):
    """Extraer datos de presión y altura de un sub-archivo para gráficos (múltiples posiciones)"""
    datos_tiempo = sub_archivo['datos']
    distancia_entre_tomas = configuracion['distancia_entre_tomas']
    
    z_datos = []
    presion_datos = []
    
    # Procesar TODAS las filas (múltiples posiciones)
    for _, fila in datos_tiempo.iterrows():
        y_traverser = fila['Y_coord']
        
        # Procesar cada toma física
        for toma_fisica in range(1, 13):
            if configuracion['orden'] == "asc":
                sensor_columna = f"Presion-Sensor {toma_fisica}"
            else:
                sensor_columna = f"Presion-Sensor {13 - toma_fisica}"
            
            if sensor_columna in datos_tiempo.columns:
                # Calcular posición Z total (altura)
                z_total = y_traverser + (toma_fisica - 1) * distancia_entre_tomas
                
                # Obtener presión
                presion = fila[sensor_columna]
                
                try:
                    if isinstance(presion, str):
                        presion = float(presion.replace(',', '.'))
                    
                    z_datos.append(z_total)
                    presion_datos.append(presion)
                    
                except:
                    continue
    
    # Ordenar por altura
    if z_datos and presion_datos:
        datos_ordenados = sorted(zip(z_datos, presion_datos))
        z_ordenado, presion_ordenada = zip(*datos_ordenados)
        return list(z_ordenado), list(presion_ordenada)
    
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
    
    # Layout CON CONFIGURACIÓN DE EJES SOLICITADA
    fig.update_layout(
        title=f"Diferencia de Perfiles: {sub_archivo_a['archivo_fuente']} - {sub_archivo_b['archivo_fuente']}",
        xaxis_title="Presión / Diferencia de Presión [Pa]",
        yaxis_title="Altura z [mm]",
        height=700, width=900,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        title_font=dict(size=16, color="#08596C"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            scaleanchor="y",      # AGREGADO: Configuración solicitada
            scaleratio=4          # AGREGADO: Configuración solicitada
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2
        )
    )
    
    return fig, area_diferencia

# Sidebar para navegación (colapsable)
with st.sidebar:
    st.markdown("### 🧭 Navegación")
    
    if st.button("🏠 Inicio", use_container_width=True):
        st.session_state.seccion_actual = 'inicio'
        st.rerun()
    
    if st.button("📊 BETZ 2D", use_container_width=True):
        st.session_state.seccion_actual = 'betz_2d'
        st.rerun()
    
    if st.button("🌪️ BETZ 3D", use_container_width=True):
        st.session_state.seccion_actual = 'betz_3d'
        st.rerun()
    
    st.divider()
    
    # Información del proyecto
    st.markdown("### ℹ️ Información")
    st.markdown(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}")
    st.markdown(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.session_state.datos_procesados:
        st.markdown(f"**Archivos procesados:** {len(st.session_state.datos_procesados)}")
    
    if st.session_state.sub_archivos_generados:
        st.markdown(f"**Sub-archivos generados:** {len(st.session_state.sub_archivos_generados)}")

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
                • Extracción de tiempo y coordenadas X,Y<br>
                • Gráficos concatenados modo Betz<br>
                • Configuración flexible de sensores
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
                • Perfiles concatenados en modo Betz<br>
                • Análisis de múltiples posiciones transversales<br>
                • Visualización 3D de campos de presión<br>
                • Interpolación espacial avanzada
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ACCEDER A BETZ 3D", key="betz_3d_btn", type="primary", use_container_width=True):
            st.session_state.seccion_actual = 'betz_3d'
            st.rerun()

elif st.session_state.seccion_actual == 'betz_2d':
    st.markdown("# 📊 BETZ 2D - Análisis Bidimensional")
    st.markdown("Análisis de perfiles de presión concatenados con extracción automática de tiempo y coordenadas")
    
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
            help="Define cómo se leen los datos de los sensores en relación a su posición física"
        )
        
        # Pregunta sobre el sensor de referencia
        st.info("🔍 **Pregunta:** ¿Qué sensor corresponde a la toma número 12 (la que se encuentra cerca del piso)?")
        sensor_referencia = st.selectbox(
            "Sensor de referencia (toma 12):",
            [f"Sensor {i}" for i in range(1, 13)],
            index=11,  # Por defecto Sensor 12
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
        
        # Guardar configuración
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
        st.image("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-KhQlQE0m6ivovNSDS28u1dNI7soVeq.png", 
                caption="Diagrama técnico de sensores",
                width=300)
    
    # Paso 2: Carga de archivos
    if st.session_state.configuracion_inicial:
        st.markdown("## 📁 Paso 2: Carga de Archivos de Incertidumbre")
        
        uploaded_files = st.file_uploader(
            "Seleccione uno o más archivos CSV de incertidumbre:",
            type=['csv'],
            accept_multiple_files=True,
            help="Los archivos deben tener el formato estándar del laboratorio con tiempo y coordenadas en el nombre"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s)")
            
            # Procesar archivos automáticamente con la configuración guardada
            for archivo in uploaded_files:
                if archivo.name not in st.session_state.datos_procesados:
                    with st.spinner(f"Procesando {archivo.name}..."):
                        # Usar la función exacta del código de referencia
                        datos = procesar_promedios(archivo, st.session_state.configuracion_inicial['orden'])
                        if datos is not None:
                            st.session_state.datos_procesados[archivo.name] = datos
                            
                            # Crear sub-archivos por tiempo y posición CON NOMBRE DE ARCHIVO
                            sub_archivos = crear_archivos_individuales_por_tiempo_y_posicion(datos, archivo.name)
                            st.session_state.sub_archivos_generados.update(sub_archivos)
                            
                            st.success(f"✅ Archivo {archivo.name} procesado correctamente")
                            st.success(f"📝 Se generaron {len(sub_archivos)} sub-archivos por tiempo y posición")
            
            # Mostrar datos procesados
            if st.session_state.datos_procesados:
                st.markdown("## 📋 Datos Procesados")
                
                for nombre_archivo, datos in st.session_state.datos_procesados.items():
                    with st.expander(f"Ver datos de {nombre_archivo}"):
                        st.dataframe(datos, use_container_width=True)
                        
                        # Mostrar tiempos y coordenadas únicos encontrados
                        tiempos_unicos = datos['Tiempo (s)'].dropna().unique()
                        coordenadas_unicas = datos[['X_coord', 'Y_coord']].drop_duplicates()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Tiempos encontrados:** {sorted(tiempos_unicos)}")
                        with col2:
                            st.markdown("**Coordenadas encontradas:**")
                            st.dataframe(coordenadas_unicas, use_container_width=True)
    
    # Paso 3: Mostrar sub-archivos generados
    if st.session_state.sub_archivos_generados:
        st.markdown("## 📂 Paso 3: Sub-archivos Generados por Archivo y Tiempo")
        
        # Mostrar resumen de sub-archivos
        st.info(f"🎯 Se generaron **{len(st.session_state.sub_archivos_generados)}** sub-archivos organizados por archivo fuente, tiempo y coordenadas")
        
        # NUEVA ORGANIZACIÓN: Agrupar por archivo fuente primero, luego por tiempo
        sub_archivos_por_fuente = {}
        for clave, sub_archivo in st.session_state.sub_archivos_generados.items():
            archivo_fuente = sub_archivo['archivo_fuente']
            tiempo = sub_archivo['tiempo']
            
            if archivo_fuente not in sub_archivos_por_fuente:
                sub_archivos_por_fuente[archivo_fuente] = {}
            if tiempo not in sub_archivos_por_fuente[archivo_fuente]:
                sub_archivos_por_fuente[archivo_fuente][tiempo] = []
            
            sub_archivos_por_fuente[archivo_fuente][tiempo].append((clave, sub_archivo))
        
        # Mostrar sub-archivos agrupados por archivo fuente
        for archivo_fuente, tiempos_dict in sub_archivos_por_fuente.items():
            with st.expander(f"📁 {archivo_fuente} - {len(tiempos_dict)} tiempos"):
                for tiempo, sub_archivos_tiempo in tiempos_dict.items():
                    # Ahora solo hay UN sub-archivo por tiempo
                    clave, sub_archivo = sub_archivos_tiempo[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**⏱️ T{tiempo}s**")
                    with col2:
                        st.markdown(f"Registros: {len(sub_archivo['datos'])}")
                    with col3:
                        # Botón para descargar sub-archivo
                        csv_data = sub_archivo['datos'].to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar CSV",
                            data=csv_data,
                            file_name=sub_archivo['nombre_archivo'],
                            mime="text/csv",
                            key=f"download_{clave}",
                            use_container_width=True
                        )
                    st.markdown("---")
        
        # Paso 4: Sección de Gráficos
        st.markdown("## 📈 Paso 4: Generación de Gráficos Concatenados")
        
        # Calcular posiciones de sensores
        posiciones_sensores = calcular_posiciones_sensores(
            st.session_state.configuracion_inicial['distancia_toma_12'],
            st.session_state.configuracion_inicial['distancia_entre_tomas'],
            st.session_state.configuracion_inicial['orden']
        )
        
        # Mostrar tabla de posiciones calculadas
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
        
        # NUEVA SELECCIÓN ORGANIZADA POR ARCHIVO FUENTE
        st.markdown("### 🎯 Selección de Sub-archivos para Gráfico Concatenado")
        st.info("💡 **Nota:** Los gráficos se agruparán por tiempo. Todos los T10s tendrán el mismo color, todos los T20s otro color, etc.")
        
        sub_archivos_seleccionados = {}
        
        # Paleta de colores para mostrar en la selección
        colores_por_tiempo = {
            10: '#08596C',   # Teal
            20: '#E74C3C',   # Rojo
            30: '#F39C12',   # Naranja
            40: '#27AE60',   # Verde
            50: '#8E44AD',   # Púrpura
            60: '#3498DB',   # Azul claro
            70: '#E67E22',   # Naranja oscuro
            80: '#2ECC71',   # Verde claro
            90: '#9B59B6',   # Púrpura claro
            100: '#1ABC9C',  # Turquesa
            110: '#F1C40F',  # Amarillo
            120: '#34495E'   # Gris azulado
        }
        
        # NUEVA ORGANIZACIÓN: Por archivo fuente primero, luego por tiempo
        for archivo_fuente in sorted(sub_archivos_por_fuente.keys()):
            st.markdown(f"## 📁 {archivo_fuente}")
            
            # Mostrar tiempos para este archivo
            for tiempo in sorted(sub_archivos_por_fuente[archivo_fuente].keys()):
                color_tiempo = colores_por_tiempo.get(tiempo, '#08596C')
                
                # Ahora solo hay UN sub-archivo por tiempo
                clave, sub_archivo = sub_archivos_por_fuente[archivo_fuente][tiempo][0]
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    incluir = st.checkbox(
                        f"⏱️ T{tiempo}s - {len(sub_archivo['datos'])} registros",
                        key=f"incluir_{clave}"
                    )
                    
                    if incluir:
                        sub_archivos_seleccionados[clave] = sub_archivo
                
                with col2:
                    st.markdown(f"""
                    <div style="background: {color_tiempo}; height: 20px; width: 60px; border-radius: 3px; margin-top: 8px;"></div>
                    """, unsafe_allow_html=True)
        
        # Generar gráfico concatenado si hay selecciones
        if sub_archivos_seleccionados:
            st.markdown("### 📊 Gráfico Concatenado Vertical Modo Betz")
            
            # Mostrar resumen de selección por tiempo
            tiempos_seleccionados = {}
            for clave, sub_archivo in sub_archivos_seleccionados.items():
                tiempo = sub_archivo['tiempo']
                if tiempo not in tiempos_seleccionados:
                    tiempos_seleccionados[tiempo] = 0
                tiempos_seleccionados[tiempo] += 1
            
            resumen_texto = ", ".join([f"T{t}s ({count} pos.)" for t, count in sorted(tiempos_seleccionados.items())])
            st.info(f"📈 Graficando: {resumen_texto}")
            
            # Crear y mostrar el gráfico concatenado
            fig = crear_grafico_betz_concatenado(
                sub_archivos_seleccionados, 
                posiciones_sensores, 
                st.session_state.configuracion_inicial
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Información adicional
                total_puntos = len(sub_archivos_seleccionados) * 12  # 12 sensores por posición
                st.success(f"✅ Gráfico vertical generado con {total_puntos} puntos de datos concatenados")
                
                # Opciones de exportación
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
                    # Crear CSV con datos concatenados
                    datos_exportar = []
                    for clave, sub_archivo in sub_archivos_seleccionados.items():
                        datos_sub = sub_archivo['datos'].copy()
                        datos_sub['Sub_archivo'] = clave
                        datos_exportar.append(datos_sub)
                    
                    df_exportar = pd.concat(datos_exportar, ignore_index=True)
                    csv_data = df_exportar.to_csv(index=False)
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
        if st.session_state.sub_archivos_generados:
            st.markdown("---")
            st.markdown("## ➖ Análisis de Diferencias de Áreas")
            st.markdown("Selecciona dos sub-archivos para calcular la diferencia de áreas entre sus perfiles de presión")
            
            # Crear lista de opciones para los selectores
            opciones_subarchivos = list(st.session_state.sub_archivos_generados.keys())
            opciones_subarchivos.sort()  # Ordenar alfabéticamente
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                archivo_a = st.selectbox(
                    "📊 Archivo A (minuendo):",
                    opciones_subarchivos,
                    key="archivo_a_resta",
                    help="Seleccione el primer sub-archivo (del cual se restará el segundo)"
                )
            
            with col2:
                st.markdown("### ➖")
                st.markdown("<div style='text-align: center; font-size: 2rem; margin-top: 1rem;'>-</div>", unsafe_allow_html=True)
            
            with col3:
                archivo_b = st.selectbox(
                    "📊 Archivo B (sustraendo):",
                    opciones_subarchivos,
                    key="archivo_b_resta",
                    help="Seleccione el segundo sub-archivo (que será restado del primero)"
                )
            
            if st.button("🔄 Calcular Diferencia de Áreas", type="primary", use_container_width=True):
                if archivo_a and archivo_b and archivo_a != archivo_b:
                    with st.spinner("Calculando diferencia de áreas..."):
                        # Crear gráfico de diferencia
                        fig_diferencia, diferencia_area = crear_grafico_diferencia_areas(
                            st.session_state.sub_archivos_generados[archivo_a],
                            st.session_state.sub_archivos_generados[archivo_b],
                            st.session_state.configuracion_inicial
                        )
                        
                        if fig_diferencia:
                            st.plotly_chart(fig_diferencia, use_container_width=True)
                            
                            # Mostrar resultados numéricos
                            col1, col2, col3 = st.columns(3)
                            
                            # Calcular áreas individuales
                            z_a, p_a = extraer_datos_para_grafico(
                                st.session_state.sub_archivos_generados[archivo_a], 
                                st.session_state.configuracion_inicial
                            )
                            z_b, p_b = extraer_datos_para_grafico(
                                st.session_state.sub_archivos_generados[archivo_b], 
                                st.session_state.configuracion_inicial
                            )
                            
                            area_a = calcular_area_bajo_curva(z_a, p_a)
                            area_b = calcular_area_bajo_curva(z_b, p_b)
                            
                            with col1:
                                st.metric(
                                    f"Área {st.session_state.sub_archivos_generados[archivo_a]['archivo_fuente']}", 
                                    f"{area_a:.2f} Pa·mm"
                                )
                            
                            with col2:
                                st.metric(
                                    f"Área {st.session_state.sub_archivos_generados[archivo_b]['archivo_fuente']}", 
                                    f"{area_b:.2f} Pa·mm"
                                )
                            
                            with col3:
                                delta_color = "normal" if diferencia_area >= 0 else "inverse"
                                st.metric(
                                    "Diferencia de Áreas", 
                                    f"{diferencia_area:.2f} Pa·mm",
                                    delta=f"{diferencia_area:.2f}",
                                    delta_color=delta_color
                                )
                            
                            # Interpretación del resultado
                            if diferencia_area > 0:
                                st.success(f"✅ El área de **{st.session_state.sub_archivos_generados[archivo_a]['archivo_fuente']}** es **{diferencia_area:.2f} Pa·mm** mayor que la de **{st.session_state.sub_archivos_generados[archivo_b]['archivo_fuente']}**")
                            elif diferencia_area < 0:
                                st.info(f"ℹ️ El área de **{st.session_state.sub_archivos_generados[archivo_b]['archivo_fuente']}** es **{abs(diferencia_area):.2f} Pa·mm** mayor que la de **{st.session_state.sub_archivos_generados[archivo_a]['archivo_fuente']}**")
                            else:
                                st.info("ℹ️ Las áreas son prácticamente iguales")
                            
                            # Botón de descarga del gráfico de diferencia
                            html_diferencia = fig_diferencia.to_html()
                            st.download_button(
                                label="📊 Descargar Gráfico de Diferencia (HTML)",
                                data=html_diferencia,
                                file_name=f"diferencia_areas_{archivo_a}_vs_{archivo_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html"
                            )
                            
                        else:
                            st.error("❌ No se pudo calcular la diferencia. Verifique que los sub-archivos tengan datos válidos.")
                else:
                    st.warning("⚠️ Seleccione dos sub-archivos diferentes para calcular la diferencia.")

elif st.session_state.seccion_actual == 'betz_3d':
    st.markdown("# 🌪️ BETZ 3D - Análisis Tridimensional")
    st.markdown("Análisis completo de perfiles de presión concatenados en modo Betz")
    
    st.info("🚧 **En Desarrollo:** La sección BETZ 3D estará disponible próximamente con funcionalidades avanzadas de análisis tridimensional.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Laboratorio de Aerodinámica y Fluidos - UTN HAEDO</strong></p>
    <p>Sistema de Análisis de Datos Aerodinámicos • Versión 2.0</p>
    <p>Desarrollado con ❤️ usando Streamlit y Python</p>
    <p><small>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</small></p>
</div>
""", unsafe_allow_html=True)
