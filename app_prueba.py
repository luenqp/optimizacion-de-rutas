print("\n====== INICIANDO APP =======\n")
import streamlit as st
import numpy as np
import pandas as pd

from datos import generar_datos
# Importar funciones del optimizador
from opti_rutas import run_optimization

if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = False
if 'costo_rutas' not in st.session_state:
    st.session_state.costo_rutas_df = pd.DataFrame()

st.write("""
## 🚚 Optimización de Rutas - Generación de Columnas
**Problema de enrutamiento de vehículos con capacidad limitada**
""")
st.divider()

# Sidebar - Configuración
with st.sidebar:
    st.header("⚙️ Configuración del Problema")
    
    st.subheader("Parámetros Principales")
    n_camiones = st.number_input(
        "Número de Camiones",
        min_value=1,
        max_value=50,
        value=5,
        help="Cantidad de camiones disponibles"
    )
    
    m_clientes = st.number_input(
        "Número de Clientes",
        min_value=1,
        max_value=26,
        value=4,
        help="Cantidad de clientes a atender"
    )
    
    st.subheader("Costos y Capacidades")
    costo_fijo = st.number_input(
        "Costo Fijo por Camión",
        min_value=0.0,
        value=2.0,
        step=0.1,
        help="Costo fijo de usar cada camión"
    )
    
    capacidad = st.number_input(
        "Capacidad de Camiones",
        min_value=0.1,
        value=10.0,
        step=0.5,
        help="Capacidad máxima de cada camión"
    )

    min_demanda = st.number_input(
        "Mínima demanda por cliente",
        min_value=0.0,
        value=0.1,
        step=0.5,
        help="Demanda MÍNIMA para el rango de valores posibles"
    )

    max_demanda = st.number_input(
        "Máxima demanda por cliente",
        min_value=min_demanda,
        value=2.0,
        step=0.5,
        help="Demanda MÁXIMA para el rango de valores posibles"
    )
    
    st.subheader("Aleatorización")
    use_seed = st.checkbox("Usar semilla aleatoria", value=True)
    seed = st.number_input(
        "Semilla",
        min_value=0,
        value=42,
        disabled=not use_seed,
        help="Para reproducibilidad de resultados"
    )
    
    st.divider()

    # Botón de optimización
    if st.button("▶️ Cargar datos", type="primary", width='stretch', key="cargar_datos"):
        datos_cargados = True
        I, J, d, Q, F, costo_rutas = generar_datos(n_camiones, m_clientes, min_demanda, max_demanda, capacidad, costo_fijo, seed)
        
        # Guardar en session_state
        st.session_state.datos_cargados = True
        st.session_state.I = I
        st.session_state.J = J
        st.session_state.d = d
        st.session_state.Q = Q
        st.session_state.F = F
        st.session_state.costo_rutas = pd.DataFrame(costo_rutas)
        st.session_state.costo_rutas_df = pd.DataFrame(costo_rutas)
        # Crear DataFrame temporal para visualización/edición
        st.session_state.demandas_df = pd.DataFrame({
        'Cliente': st.session_state.J,
        'Demanda': st.session_state.d
        })
        
        st.success("✅ Datos cargados correctamente")

st.markdown("### 🛢Datos del Problema")
# Tabs para diferentes vistas
tab1, tab2, tab3 = st.tabs(["General", "Demandas", "Costos de rutas"])

if st.session_state.datos_cargados:
    with tab1:
        st.write("Contenido de General")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Camiones", len(st.session_state.I))
        with col2:
            st.metric("Total Clientes", len(st.session_state.J))
        st.write(f"Camiones (I): {st.session_state.I}")
        st.write(f"Clientes (J): {st.session_state.J}")

    with tab2:
        st.write("Contenido de Demandas")
        # Editor de demandas
        edited_demandas = st.data_editor(
            st.session_state.demandas_df,
            column_config={
                "Cliente": st.column_config.NumberColumn("Cliente", disabled=True),
                "Demanda": st.column_config.NumberColumn(
                    "Demanda",
                    min_value=0.0,
                    format="%.1f")},
            hide_index=True,
            width='stretch',
            key="demandas_editor"
            )
    
        # Convertir de vuelta a numpy array y actualizar session_state
        # Solo actualizar si hubo cambios
        new_demandas = edited_demandas['Demanda'].to_numpy()
        if not np.array_equal(new_demandas, st.session_state.d):
            st.session_state.d = new_demandas

    with tab3:
        st.write("Matriz de costos de rutas")
        edited_df = st.data_editor(st.session_state.costo_rutas_df, key="costos_editor")
        if not edited_df.equals(st.session_state.costo_rutas_df):
            st.session_state.costo_rutas = edited_df
        st.info(f"📊 Total de rutas: {type(edited_df)}"); print("\nedited")

st.divider()
## Ejecutamos optimización
if st.button("Ejecutar optimización", type="primary", use_container_width=True, disabled=st.session_state.datos_cargados==False):
    resultados = run_optimization(st.session_state.I, st.session_state.J, st.session_state.F, st.session_state.Q, st.session_state.d, n_camiones, m_clientes, costo_fijo, capacidad, st.session_state.costo_rutas.to_numpy(), seed)

    st.subheader("📊 Resultados de la Optimización")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="💰 Costo Total Óptimo",
            value=f"${resultados['z_optimal']:.3f}"
        )
    
    with col2:
        st.metric(
            label="🚚 Camiones Utilizados",
            value=len(resultados['routes'])
        )
    
    with col3:
        st.metric(
            label="🔄 Iteraciones CG",
            value=len(resultados['iterations'])
        )
    
    with col4:
        st.metric(
            label="⏱️ Tiempo de Ejecución",
            value=f"{resultados['time_minutes']:.3f} min"
        )
    
    st.divider()

    st.dataframe(resultados['routes'])

