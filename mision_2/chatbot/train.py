# train.py
from __future__ import annotations
from typing import List, Tuple, Dict
from pathlib import Path
import json, re
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Utilidades ----------
def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[áàä]", "a", t)
    t = re.sub(r"[éèë]", "e", t)
    t = re.sub(r"[íìï]", "i", t)
    t = re.sub(r"[óòö]", "o", t)
    t = re.sub(r"[úùü]", "u", t)
    return t

def load_data(path: str = "intents.json") -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs: List[Tuple[str, str]] = []
    responses: Dict[str, List[str]] = {}
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            pairs.append((pattern, tag))
        responses[tag] = intent["responses"]
    return pairs, responses

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

# ---------- Entrenamiento ----------
def main() -> None:
    pairs, responses = load_data("intents.json")
    X = [normalize(t) for t, _ in pairs]
    y = [intent for _, intent in pairs]

    pipe = build_pipeline()
    pipe.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    dump(pipe, "models/model.pkl")
    with open("models/responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print("✅ Modelo guardado en models/model.pkl")
    print("✅ Respuestas guardadas en models/responses.json")

if __name__ == "__main__":
    main()
