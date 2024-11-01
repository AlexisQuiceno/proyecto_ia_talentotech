import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Cargar los datos del chatbot
lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25  # umbral de confianza mínima
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return [classes[i[0]] for i in results] if results else ["no_entiendo"]



# Obtenemos una respuesta aleatoria
def get_response(tags, intents_json):
    if "no_entiendo" in tags:
        return "Lo siento, no entiendo esa pregunta. ¿Podrías reformularla o intentar con otra consulta de temas legales?"

    responses = []
    for tag in tags:
        for intent in intents_json['intents']:
            if intent["tag"] == tag:
                responses.append(random.choice(intent['responses']))
                break
    return " ".join(responses)



# Función para obtener la respuesta del chatbot
def respuesta(message):
    tags = predict_class(message)
    res = get_response(tags, intents)
    return res

# Función para manejar el evento de enviar mensaje
def send_message():
    user_input = user_entry.get("1.0", tk.END).strip()
    chat_window.insert(tk.END, f"Tú: {user_input}\n")
    response = respuesta(user_input)  # Llamamos a la función respuesta
    chat_window.insert(tk.END, f"Bot: {response}\n\n")
    user_entry.delete("1.0", tk.END)

# Configuración de la ventana principal de la interfaz
window = tk.Tk()
window.title("Asistente Legal Chatbot")
window.geometry("500x550")
window.config(bg="lightblue")

# Crear un área de texto desplazable para mostrar el chat
chat_window = scrolledtext.ScrolledText(window, width=60, height=20, wrap=tk.WORD, state="normal", bg="white")
chat_window.pack(padx=10, pady=10)
chat_window.insert(tk.END, "Bot: Hola, soy tu asistente legal. ¿En qué puedo ayudarte hoy?\n\n")

# Crear un cuadro de texto para que el usuario escriba sus preguntas
user_entry = tk.Text(window, height=2, width=50, wrap=tk.WORD)
user_entry.pack(padx=10, pady=5)

# Botón para enviar el mensaje
send_button = tk.Button(window, text="Enviar", width=10, command=send_message)
send_button.pack(pady=5)

# Ejecutar la ventana principal
window.mainloop()
