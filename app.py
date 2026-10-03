import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Aprobación de Curso",
    page_icon="🎓",
    layout="wide"
)

# Estilo CSS personalizado para mejorar la visualización
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Predicción de Aprobación de Curso</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Cargue un archivo de Excel o ingrese los datos manualmente para predecir la nota estimada usando Inteligencia Artificial</div>', unsafe_allow_html=True)

# Barra lateral con información útil
with st.sidebar:
    st.header("⚙️ Recursos del Modelo")
    st.info("Este sistema utiliza un modelo optimizado de Bagging junto con preprocesadores calibrados para estimar el desempeño de los estudiantes.")
    st.write("**Archivos requeridos:**")
    st.markdown("""
    - `one_hot_columns.joblib`
    - `min_max_scaler.joblib`
    - `bagging_optimizado.joblib`
    """)

# Función para procesar y preparar los datos (ya sea individuales o masivos)
def procesar_datos(df_raw):
    df_procesado = df_raw.copy()
    
    # 1. Eliminar variables innecesarias si están presentes
    columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
    df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
    
    # 2. Cargar estructura de columnas esperadas para One-Hot
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    
    # Asegurar codificación One-Hot de Felder
    if 'Felder' in df_procesado.columns:
        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'].astype(str).str.strip().str.lower() == valor_esperado.lower()).astype(float)
        else:
            df_encoded = pd.get_dummies(df_procesado['Felder'])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]
            
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
    else:
        # Si no viene la columna Felder, se rellenan sus categorías en 0
        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col in si_columnas_one_hot:
                df_procesado[col] = 0.0
                
    # 3. Escalar Examen de Admisión
    if 'Examen_admisión' in df_procesado.columns:
        scaler = joblib.load('min_max_scaler.joblib')
        # Escalar valores usando el min_max_scaler pre-entrenado
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')
    else:
        df_procesado['Examen_admision_scaled'] = 0.0
        
    # Reordenar columnas para coincidir exactamente con el orden original del modelo
    if isinstance(one_hot_transformer, list):
        df_procesado = df_procesado.reindex(columns=one_hot_transformer, fill_value=0.0)
        
    return df_procesado

# Diseño en pestañas para alternar entre predicción individual o masiva
tab1, tab2 = st.tabs(["👤 Predicción Individual", "📁 Predicción Masiva (Subir Excel)"])

with tab1:
    st.header("Ingrese los datos del estudiante")
    
    col1, col2 = st.columns(2)
    with col1:
        opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
        felder_input = st.selectbox("Estilo de aprendizaje (Felder):", opciones_felder)
    
    with col2:
        examen_input = st.number_input("Puntaje Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.8, step=0.01)
        
    if st.button("🎯 Calcular Nota Estimada", use_container_width=True):
        try:
            # Construir DataFrame temporal
            df_input = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
            df_final = procesar_datos(df_input)
            
            # Cargar modelo y predecir
            model = joblib.load('bagging_optimizado.joblib')
            prediccion = model.predict(df_final)[0]
            
            # Mostrar resultados de forma visual e interactiva
            st.success("¡Predicción generada con éxito!")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Nota Final Estimada", value=f"{prediccion:.3f}")
            with c2:
                estado = "Aprobado (>= 3.0)" if prediccion >= 3.0 else "Reprobado (< 3.0)"
                color = "green" if prediccion >= 3.0 else "red"
                st.markdown(f"### Estado Estimado: <span style='color:{color}'>{estado}</span>", unsafe_allow_html=True)
                
            with st.expander("Ver datos procesados para el modelo"):
                st.dataframe(df_final)
                
        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")

with tab2:
    st.header("Predicción masiva desde archivo Excel")
    st.write("Suba un archivo Excel que contenga al menos las columnas `Felder` y `Examen_admisión`.")
    
    uploaded_file = st.file_uploader("Seleccione el archivo .xlsx", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Leer archivo de entrada
            df_subido = pd.read_excel(uploaded_file)
            st.subheader("Vista previa de los datos subidos")
            st.dataframe(df_subido.head(10))
            
            if st.button("🚀 Procesar y Predecir Archivo Completo", use_container_width=True):
                with st.spinner("Procesando registros..."):
                    # Procesar los datos subidos
                    df_procesado_masivo = procesar_datos(df_subido)
                    
                    # Realizar la predicción masiva
                    model = joblib.load('bagging_optimizado.joblib')
                    predicciones = model.predict(df_procesado_masivo)
                    
                    # Añadir predicciones al DataFrame original
                    df_resultado = df_subido.copy()
                    df_resultado['Nota_final_estimada'] = predicciones
                    df_resultado['Aprobación_estimada'] = np.where(predicciones >= 3.0, 'si', 'no')
                    
                    st.success("¡Procesamiento masivo completado!")
                    st.subheader("Resultados de la Predicción")
                    st.dataframe(df_resultado)
                    
                    # Crear botón de descarga para los resultados
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_resultado.to_excel(writer, index=False, sheet_name='Predicciones')
                    processed_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Descargar Resultados en Excel",
                        data=processed_data,
                        file_name="predicciones_curso.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo Excel: {e}")
