import streamlit as st
import joblib
import pandas as pd
import numpy as np
from groq import Groq
import tensorflow as tf
import json # <-- Añadido para procesar nuestro diccionario

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS (UI)
# ==========================================
st.set_page_config(page_title="MadriDeep AI", page_icon="🏢", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    .big-title { font-size:40px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .sub-title { font-size:20px !important; color: #6B7280; text-align: center; margin-bottom: 30px; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE RECURSOS (Deep Learning)
# ==========================================
@st.cache_resource
def cargar_recursos():
    try:
        prep = joblib.load('preprocessor.joblib')
        mod_buy = tf.keras.models.load_model('modelo_compra_definitivo.keras')
        mod_rent = tf.keras.models.load_model('modelo_alquiler_dl.keras')
        return prep, mod_buy, mod_rent, True
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, False

preprocessor, m_buy, m_rent, modelos_listos = cargar_recursos()

# Instanciamos el cliente de Groq una sola vez
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# ==========================================
# 3. LÓGICA DE IA: CHATBOT CON GUARDRAILS
# ==========================================
def hablar_con_ia(mensaje_usuario, tipo_operacion):
    try:
        # AQUI ESTÁ EL GUARDRAIL (Barreras de seguridad estrictas)
        instrucciones = (
            f"Eres 'MadriDeep', un experto inmobiliario EXCLUSIVO de la Comunidad de Madrid. "
            f"El usuario busca información sobre {tipo_operacion}. "
            "REGLAS ESTRICTAS: "
            "1. SOLO puedes hablar sobre el mercado inmobiliario, alquiler, compra, barrios y tasaciones en Madrid. "
            "2. Si el usuario te pregunta sobre programación, cocina, historia, otros países o CUALQUIER tema no relacionado con inmuebles en Madrid, "
            "DEBES NEGARTE EDUCADAMENTE diciendo que eres un asistente inmobiliario y no puedes responder a eso. "
            "3. Sé profesional, directo y usa datos realistas."
        )
        
        mensajes_ia = [{"role": "system", "content": instrucciones}]
        for m in st.session_state.messages:
            mensajes_ia.append({"role": m["role"], "content": m["content"]})
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})
        
        completion = client.chat.completions.create(
            messages=mensajes_ia,
            model="llama-3.3-70b-versatile",
            temperature=0.3 # Bajamos un poco la temperatura para que sea más estricto con las reglas
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error técnico: {str(e)}"

# ==========================================
# 4. LÓGICA DE IA: EXTRACCIÓN Y PREDICCIÓN (Nuestro Pipeline)
# ==========================================
def extraer_y_predecir(texto_anuncio):
    prompt_sistema = """
    Eres un analista de datos inmobiliarios. Tu primera tarea es determinar si el texto del usuario
    es un anuncio o descripción de un inmueble en venta o alquiler.

    PRIMERO, añade siempre esta clave al JSON:
    - "es_anuncio_inmobiliario": (booleano) true si el texto describe un inmueble, false si es cualquier
      otra cosa (saludos, preguntas, texto sin sentido, etc.).

    Si "es_anuncio_inmobiliario" es false, devuelve SOLO ese campo y nada más.

    Si "es_anuncio_inmobiliario" es true, extrae también estas claves EXACTAS
    (usa null si el texto no menciona el dato):
    - "sq_mt_built": (número) Metros cuadrados.
    - "n_rooms": (número) Número de habitaciones.
    - "n_bathrooms": (número) Número de baños.
    - "floor": (texto o número) Planta.
    - "house_type_id": (texto) Tipo de vivienda.
    - "Distrito": (texto) Distrito de la ciudad.
    - "neighborhood_id": (texto) Barrio.
    - "subtitle": (texto) Título corto.
    - "energy_certificate": (texto) Certificado energético.
    - "is_exterior": (booleano) ¿Es exterior? (Por defecto false si no lo dice).
    - "is_renewal_needed": (booleano) ¿Necesita reforma?
    - "is_floor_under": (booleano) ¿Es un bajo?
    - "has_lift": (booleano) ¿Tiene ascensor?
    - "has_parking": (booleano) ¿Tiene garaje o parking?
    - "is_new_development": (booleano) ¿Es obra nueva?
    """
    
    try:
        # Usamos Groq para extraer el JSON rapidísimo
        respuesta = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": texto_anuncio}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        diccionario_ia = json.loads(respuesta.choices[0].message.content)

        # --- GUARDRAIL: verificar que es un anuncio inmobiliario real ---
        if not diccionario_ia.get("es_anuncio_inmobiliario", True):
            return "NO_ES_ANUNCIO", None, None

        # --- RED DE SEGURIDAD MEJORADA ---
        columnas_exigidas = ['sq_mt_built', 'n_rooms', 'neighborhood_id', 'is_exterior', 'Distrito', 'is_renewal_needed', 'subtitle', 'is_floor_under', 'has_lift', 'is_new_development', 'energy_certificate', 'has_parking', 'floor', 'n_bathrooms', 'house_type_id']
        
        # Valores por defecto válidos para cada columna (basados en las categorías del preprocesador)
        DEFAULTS = {
            'sq_mt_built':        80.0,
            'n_rooms':            3.0,
            'n_bathrooms':        1.0,
            'floor':              1.0,      # StandardScaler → valor numérico
            'is_exterior':        False,
            'is_renewal_needed':  False,
            'is_floor_under':     False,
            'has_lift':           False,
            'has_parking':        False,
            'is_new_development': False,
            'house_type_id':      'Pisos',          # Categoría más común
            'energy_certificate': 'other',           # Categoría comodín válida
            'subtitle':           'Centro',          # Barrio válido genérico
            'Distrito':           'Centro',          # Distrito válido genérico
            'neighborhood_id':    'Centro',          # Barrio válido genérico
        }

        diccionario_seguro = {}
        for col in columnas_exigidas:
            valor = diccionario_ia.get(col)
            if valor is None:
                diccionario_seguro[col] = DEFAULTS.get(col, None)
            else:
                # Normalizar floor a número si la IA lo devuelve como texto (ej: "3ª", "bajo")
                if col == 'floor':
                    try:
                        diccionario_seguro[col] = float(str(valor).replace('ª','').replace('º','').strip())
                    except (ValueError, TypeError):
                        diccionario_seguro[col] = DEFAULTS['floor']
                else:
                    diccionario_seguro[col] = valor
        # ---------------------------------
        
        # Predicción
        df_para_predecir = pd.DataFrame([diccionario_seguro])
        datos_procesados = preprocessor.transform(df_para_predecir)
        
        # Keras devuelve arrays 2D tipo [[valor]], por eso usamos [0][0]
        pred_compra_log = m_buy.predict(datos_procesados, verbose=0)[0][0]
        precio_compra = np.exp(pred_compra_log) # Asumiendo que compra usa logaritmo
        
        precio_alquiler = m_rent.predict(datos_procesados, verbose=0)[0][0] # Asumiendo que alquiler no usa logaritmo
        
        return diccionario_seguro, precio_compra, precio_alquiler
        
    except Exception as e:
        st.error(f"Detalle técnico del error: {e}") 
        return None, None, None

# ==========================================
# 5. BARRA LATERAL
# ==========================================
with st.sidebar:
    st.title("🏢 MadriDeep AI")
    st.write("---")
    if modelos_listos:
        st.success("✅ Motores Activos")
    else:
        st.error("❌ Error de carga de modelos")
    
    st.write("---")
    if st.button("🔄 Borrar Historial", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 6. DISEÑO PRINCIPAL (PESTAÑAS)
# ==========================================
st.markdown('<p class="big-title">MadriDeep AI</p>', unsafe_allow_html=True)

# Creamos dos pestañas para separar responsabilidades
tab1, tab2 = st.tabs(["💬 Chat Asesor", "📊 Tasador Automático de Anuncios"])

# --- PESTAÑA 1: EL CHAT DE ROCÍO ---
with tab1:
    st.markdown('<p class="sub-title">Tu asesor inmobiliario personal</p>', unsafe_allow_html=True)
    modo = st.radio("Contexto del chat:", ["Compra 💰", "Alquiler 🔑"], horizontal=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ej: ¿Qué barrio es mejor para familias?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = hablar_con_ia(prompt, modo)
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})


# --- PESTAÑA 2: NUESTRO MOTOR DE PREDICCIÓN ---
with tab2:
    st.markdown('<p class="sub-title">Pega un anuncio y calcularemos su valor real</p>', unsafe_allow_html=True)
    
    texto_anuncio = st.text_area("Texto del anuncio (Ej: Idealista, Fotocasa...):", height=150, 
                                 placeholder="Ej: Precioso piso de 90m2 en el barrio de Salamanca, 3 habitaciones, exterior con ascensor...")
    
    if st.button("🚀 Analizar y Tasar Inmueble", type="primary"):
        if texto_anuncio.strip() == "":
            st.warning("Por favor, pega el texto de un anuncio primero.")
        else:
            with st.spinner("Extrayendo datos con IA y ejecutando modelos Deep Learning..."):
                datos_extraidos, p_compra, p_alquiler = extraer_y_predecir(texto_anuncio)

                if datos_extraidos == "NO_ES_ANUNCIO":
                    st.warning(
                        "⚠️ **Este tasador solo analiza anuncios inmobiliarios.** "
                        "Por favor, pega el texto de un anuncio de venta o alquiler "
                        "(de Idealista, Fotocasa, Habitaclia, etc.) y lo tasaremos al instante."
                    )
                elif datos_extraidos is not None:
                    st.success("¡Análisis completado con éxito!")

                    # Mostramos los precios en grande
                    col1, col2 = st.columns(2)
                    col1.metric("💰 Precio Justo de Compra", f"{p_compra:,.0f} €")
                    col2.metric("📅 Precio Justo de Alquiler", f"{p_alquiler:,.0f} € / mes")

                    # Mostramos los datos que la IA entendió (Transparencia)
                    st.write("---")
                    st.write("📋 **Datos interpretados por la IA:**")
                    st.json(datos_extraidos)
                else:
                    st.error("Hubo un problema procesando el anuncio. Intenta con otro texto.")
