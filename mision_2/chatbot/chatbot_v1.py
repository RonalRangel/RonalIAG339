# chatbot_v1.py
from __future__ import annotations
from typing import List, Tuple, Dict
import re
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import json  # <--- agrégalo junto a los demás imports
import random


# Cargar dataset desde intents.json
def load_data(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    responses = {}
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            pairs.append((pattern, tag))
        responses[tag] = intent["responses"]
    return pairs, responses

# Dataset mínimo (texto, intent)
DATA: List[Tuple[str, str]] = [
    ("hola", "saludo"),
    ("buenos dias", "saludo"),
    ("buenas tardes", "saludo"),
    ("como estas", "saludo"),
    ("adios", "despedida"),
    ("hasta luego", "despedida"),
    ("nos vemos", "despedida"),
    ("gracias", "agradecimiento"),
    ("precio del producto", "precio"),
    ("cuanto cuesta", "precio"),
    ("horarios de atencion", "horario"),
    ("cuando abren", "horario"),
    ("direccion de la tienda", "ubicacion"),
    ("donde estan ubicados", "ubicacion"),
]

RESPUESTAS: Dict[str, str] = {
    "saludo": "¡Hola! ¿En qué puedo ayudarte?",
    "despedida": "¡Hasta luego!",
    "agradecimiento": "¡Con gusto! ¿Algo más?",
    "precio": "Depende del producto. ¿Cuál te interesa?",
    "horario": "Atendemos de L-V 9:00–18:00.",
    "ubicacion": "Estamos en Calle Falsa 123.",
    "__fallback__": "Perdón, no entendí bien. ¿Puedes reformular?",
}

def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[áàä]", "a", t)
    t = re.sub(r"[éèë]", "e", t)
    t = re.sub(r"[íìï]", "i", t)
    t = re.sub(r"[óòö]", "o", t)
    t = re.sub(r"[úùü]", "u", t)
    return t

def train_bot(pairs: List[Tuple[str, str]]) -> Pipeline:
    X = [normalize(t) for t, _ in pairs]
    y = [intent for _, intent in pairs]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),

    ])
    pipe.fit(X, y)
    return pipe

def reply(model: Pipeline, text: str) -> str:
    intent = model.predict([normalize(text)])[0]
    return RESPUESTAS.get(intent, RESPUESTAS["__fallback__"])

def main() -> None:
    data, responses = load_data()  # ← ahora lee el JSON
    model = train_bot(data)
    print("Chatbot supervisado v1 listo. Escribe 'salir' para terminar.")
    while True:
        user = input("> ").strip()
        if user.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        intent = model.predict([normalize(user)])[0]
        opciones = responses.get(intent, ["Perdón, no entendí."])
        print(random.choice(opciones))


if __name__ == "__main__":
    main()
