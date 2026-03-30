import streamlit as st
import joblib
import pandas as pd
import numpy as np
from groq import Groq
import tensorflow as tf

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS (UI)
# ==========================================
st.set_page_config(page_title="MadriDeep AI", page_icon="🏢", layout="wide")

# Estilo CSS para que se vea profesional
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
    except:
        return None, None, None, False

preprocessor, m_buy, m_rent, modelos_listos = cargar_recursos()

# ==========================================
# 3. LÓGICA DE IA (Groq Cloud)
# ==========================================
def hablar_con_ia(mensaje_usuario, tipo_operacion):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
        
        instrucciones = (
            f"Eres 'MadriDeep', un experto inmobiliario en Madrid. El usuario busca {tipo_operacion}. "
            "Usa el historial para no repetir preguntas. Sé profesional y da datos reales de Madrid."
        )
        
        mensajes_ia = [{"role": "system", "content": instrucciones}]
        for m in st.session_state.messages:
            mensajes_ia.append({"role": m["role"], "content": m["content"]})
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})
        
        # AQUÍ ESTÁ LA CLAVE: completions (con S)
        completion = client.chat.completions.create(
            messages=mensajes_ia,
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error técnico: {str(e)}"

# ==========================================
# 4. BARRA LATERAL
# ==========================================
with st.sidebar:
    st.title("🏢 MadriDeep AI")
    modo = st.radio("Operación:", ["Compra 💰", "Alquiler 🔑"])
    st.write("---")
    if modelos_listos:
        st.success("✅ Motores DL Activos")
    else:
        st.error("❌ Erro de carga de modelos")
    
    st.write("---")
    if st.button("🔄 Nueva Búsqueda", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 5. DISEÑO PRINCIPAL Y CHAT
# ==========================================
st.markdown('<p class="big-title">MadriDeep AI: Tu Asesor en Madrid</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">Análisis inteligente para {modo.lower()}</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿En qué barrio buscamos hoy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando mercado..."):
            respuesta = hablar_con_ia(prompt, modo)
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
