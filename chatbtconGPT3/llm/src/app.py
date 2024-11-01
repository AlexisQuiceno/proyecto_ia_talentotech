import openai
from dotenv import load_dotenv
import os
import streamlit as st

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("openai.api_key")

# Título de la aplicación
st.set_page_config(page_title="Asesor Jurídico", page_icon="⚖️")
st.title("Asesor Jurídico Virtual ⚖️")
st.write("Bienvenido a tu asistente jurídico. Realiza consultas legales y resuelve tus dudas en tiempo real.")

# Configuración inicial de mensajes en la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola, estoy aquí para ayudarte con tus procesos jurídicos."}]

# Mostrar historial de mensajes
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['content']}**" if msg["role"] == "assistant" else msg["content"])

# Input y respuesta del modelo de OpenAI
if user_input := st.chat_input("Escribe tu consulta aquí..."):
    # Agregar mensaje del usuario
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Solicitar respuesta del modelo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state["messages"]
    )
    response_message = response['choices'][0]['message']['content']

    # Agregar respuesta del asistente
    st.session_state["messages"].append({"role": "assistant", "content": response_message})
    st.chat_message("assistant").write(response_message)
