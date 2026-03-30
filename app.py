import streamlit as st
import joblib
import pd as pd
import numpy as np
from groq import Groq
import tensorflow as tf

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS (UI)
# ==========================================
st.set_page_config(page_title="MadriDeep AI", page_icon="🏢", layout="wide")

# Estilo CSS personalizado para hacerla más "bonita"
st.markdown("""
    <style>
    /* Cambiar el fondo de la barra lateral */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    /* Estilo para los títulos */
    .big-title {
        font-size:40px !important;
        font-weight: bold;
        color: #1E3A8A; /* Azul oscuro corporativo */
        text-align: center;
        margin-bottom: 20px;
    }
    /* Estilo para los subtítulos */
    .sub-title {
        font-size:20px !important;
        color: #6B7280; /* Gris */
        text-align: center;
        margin-bottom: 30px;
    }
    /* Tarjetas de información */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
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
        return None, None, None, False

preprocessor, m_buy, m_rent, modelos_listos = cargar_recursos()


# ==========================================
# 3. LÓGICA DE INTELIGENCIA ARTIFICIAL (Groq Cloud)
# ==========================================
def hablar_con_ia(mensaje_usuario, tipo_operacion):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
        
        # System Prompt optimizado para ser un experto con memoria
        instrucciones = (
            f"Eres 'MadriDeep', un asesor inmobiliario de élite en Madrid. El usuario busca {tipo_operacion}. "
            "REGLAS CORTAS: 1. Usa el historial para NO REPETIR preguntas sobre barrio, habitaciones o precio. "
            "2. Sé profesional pero muy cercano. 3. Da opiniones de mercado reales (ej: si 800€ en Sol es realista)."
        )
        
        # Construimos la memoria (System + Historial + Nuevo mensaje)
        mensajes_ia = [{"role": "system", "content": instrucciones}]
        for m in st.session_state.messages:
            mensajes_ia.append({"role": m["role"], "content": m["content"]})
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})
        
        # Llamada a la IA con el modelo avanzado
        completion = client.chat.completions.create(
            messages=mensajes_ia,
            model="llama-3.3-70b-versatile",
            temperature=0.6
        )
        return completion.choices[0].message.content

    except KeyError:
        return "⚠️ Error: No se encontró la clave API en la configuración."
    except Exception as e:
        return f"Ocurrió un pequeño error técnico: {str(e)}"


# ==========================================
# 4. BARRA LATERAL (Panel de Control Avanzado)
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/44/Coat_of_Arms_of_Madrid.svg", width=100) # Escudo de Madrid opcional
    st.title("MadriDeep AI")
    st.subheader("Configuración")
    
    modo = st.radio("Selleciona tu operación:", ["Compra 💰", "Alquiler 🔑"], help="Cambia entre buscar para comprar o alquilar.")
    
    st.write("---")
    
    # INDICADOR DE ESTADO DE LOS MODELOS (Más estético)
    if modelos_listos:
        st.success("✅ Motores DL Conectados")
        st.caption("Los modelos de Deep Learning están activos y listos.")
    else:
        st.error("❌ Error de Conexión")
        st.caption("No se pudieron cargar los archivos de IA (.keras).")

    st.write("---")

    # LÓGICA DEL BOTÓN DE NUEVA BÚSQUEDA
    st.subheader("Acciones")
    if st.button("🔄 Empezar Nueva Búsqueda", help="Borra la conversación actual para empezar de cero.", use_container_width=True):
        st.session_state.messages = [] # Borramos el historial
        st.rerun() # Recargamos la app


# ==========================================
# 5. CUERPO PRINCIPAL (Diseño y Chat)
# ==========================================

# Título Principal (con estilo CSS)
st.markdown('<p class="big-title">🏢 MadriDeep AI: Tu Asesor en Madrid</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">Analizando oportunidades reales de {modo.lower()} en el mercado madrileño.</p>', unsafe_allow_html=True)

# Contenedor para el chat (para que no esté suelto)
chat_container = st.container()

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores dentro del contenedor
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Diseño avanzado para las respuestas de la IA
            if message["role"] == "assistant":
                st.markdown(f"""
                <div style="background-color: #f9fafb; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A;">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("Ej: Busco un piso en Sol con 2 habitaciones por 300.000€..."):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generar respuesta de la IA (con indicador de carga)
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Consultando al experto madrileño..."): # Efecto de carga
                respuesta = hablar_con_ia(prompt, modo)
                st.markdown(f"""
                <div style="background-color: #f9fafb; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A;">
                    {respuesta}
                </div>
                """, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
