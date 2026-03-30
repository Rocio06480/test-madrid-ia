import streamlit as st
import joblib
import pandas as pd
import numpy as np
from groq import Groq
import tensorflow as tf

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="IA Inmobiliaria Madrid", page_icon="🏠")

# 2. CARGA DE MODELOS (Versión Deep Learning - Keras)
@st.cache_resource
def cargar_recursos():
    try:
        # Carga del preprocesador (transformación de datos)
        prep = joblib.load('preprocessor.joblib')
        
        # Carga de redes neuronales (.keras)
        mod_buy = tf.keras.models.load_model('modelo_compra_definitivo.keras')
        mod_rent = tf.keras.models.load_model('modelo_alquiler_dl.keras')
        
        return prep, mod_buy, mod_rent, True
    except Exception as e:
        return None, None, None, False

preprocessor, m_buy, m_rent, modelos_listos = cargar_recursos()

# 3. LÓGICA DE INTELIGENCIA ARTIFICIAL (Groq Cloud)
def hablar_con_ia(mensaje_usuario, tipo_operacion):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
        
        # 1. Creamos las instrucciones de comportamiento (System Prompt)
        instrucciones = (
            f"Eres un asesor inmobiliario experto en Madrid. El usuario busca {tipo_operacion}. "
            "REGLAS: 1. Lee el historial de mensajes. Si el usuario ya dijo el barrio (ej. Sol), habitaciones o precio, ¡NO lo preguntes de nuevo! "
            "2. Sé amable, profesional y directo. 3. Si ya tienes los datos, da tu opinión experta sobre el mercado."
        )
        
        # 2. Construimos la MEMORIA (Historial + Mensaje nuevo)
        mensajes_ia = [{"role": "system", "content": instrucciones}]
        
        # Añadimos los mensajes que ya han pasado en el chat
        for m in st.session_state.messages:
            mensajes_ia.append({"role": m["role"], "content": m["content"]})
            
        # Añadimos la pregunta que acaba de hacer el usuario
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})
        
        # 3. Llamamos a la IA (usando el modelo más estable)
        completion = client.chat.completion.create(
            messages=mensajes_ia,
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
        return completion.choices[0].message.content

    except KeyError:
        return "⚠️ Error: No se encontró la 'GROQ_API_KEY' en los Secrets."
    except Exception as e:
        return f"Ocurrió un pequeño error técnico: {str(e)}"

# 4. DISEÑO DE LA INTERFAZ (Sidebar)
with st.sidebar:
    st.header("Panel de Control")
    modo = st.radio("Selecciona operación:", ["Compra 💰", "Alquiler 🔑"])
    st.write("---")
    
    if modelos_listos:
        st.success("Sistemas conectados ")
        st.caption("Motores Deep Learning activos.")
    else:
        st.error("Error: Archivos de modelo no encontrados.")

# 5. CUERPO PRINCIPAL Y CHAT
st.title("Asistente Inmobiliario Inteligente")
st.info(f"Actualmente analizando el mercado de **{modo.lower()}** en Madrid.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("Ej: Busco un piso en Chamberí con 2 habitaciones..."):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta de la IA
    with st.chat_message("assistant"):
        respuesta = hablar_con_ia(prompt, modo)
        st.markdown(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
