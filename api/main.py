from src.model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    save_data,
    load_data,
)
import argparse
from fastapi import FastAPI
from elasticsearch import Elasticsearch

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# ==========================
# Définition des arguments
# ==========================
parser = argparse.ArgumentParser(description="Pipeline de Modèle")
parser.add_argument("--prepare", action="store_true", help="Préparation des données")
parser.add_argument("--train", action="store_true", help="Entraînement du modèle")
parser.add_argument("--evaluate", action="store_true", help="Évaluation du modèle")
parser.add_argument("--save", action="store_true", help="Sauvegarde du modèle")
parser.add_argument("--load", action="store_true", help="Chargement du modèle")
args = parser.parse_args()

# ==========================
# Préparation des données
# ==========================
if args.prepare:
    print("📌 Préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
    save_data(X_train, X_test, y_train, y_test)
    print("✅ Données préparées et sauvegardées.")

# ==========================
# Entraînement du modèle
# ==========================
if args.train:
    print("📌 Entraînement du modèle...")
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    print("✅ Modèle entraîné.")

# ==========================
# Évaluation du modèle
# ==========================
if args.evaluate:
    print("📌 Évaluation du modèle...")
    try:
        # Essayer de charger le modèle depuis un fichier
        model = load_model()
    except Exception:  # Pas besoin de capturer l'exception si on ne l'utilise pas
        print("⚠️ Modèle non trouvé, entraînement en cours...")
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
    # Évaluation
    X_train, X_test, y_train, y_test = load_data()
    evaluate_model(model, X_test, y_test)

if args.save:
    print("📌 Sauvegarde du modèle...")
    try:
        model = load_model()
    except Exception:
        print("⚠️ Modèle non trouvé, entraînement en cours...")
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)

    save_model(model)
    print("✅ Modèle sauvegardé avec succès.")
# ==========================
# Chargement et évaluation
# ==========================
if args.load:
    print("📌 Chargement du modèle et évaluation...")
    model = load_model()
    X_train, X_test, y_train, y_test = load_data()
    evaluate_model(model, X_test, y_test)
    print("✅ Modèle chargé et évalué.")
