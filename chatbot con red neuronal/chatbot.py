import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()

#Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json').read()) 
words = pickle.load(open('words.pkl', 'rb')) 
classes = pickle.load(open('classes.pkl', 'rb')) 
model = load_model('chatbot_model.h5') 

#Pasamos las palabras de oración a su forma raíz 
def clean_up_sentence(sentence): 
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] 
    return sentence_words 

#Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    print(bag)
    return np.array(bag)

#Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25  # umbral de confianza mínima
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        return classes[results[0][0]]
    else:
        return "No entiendo esa pregunta"  # Mensaje de error si no hay respuesta adecuada

#Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

def respuesta(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

while True:
    message = input()
    print(respuesta(message))
