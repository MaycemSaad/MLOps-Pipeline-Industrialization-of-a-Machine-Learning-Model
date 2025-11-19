from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_pipeline import prepare_data, train_model, save_model, load_model
from fastapi.responses import FileResponse
import csv
import pandas as pd
import threading
import os

# =============================
# 🔹 Initialisation de l'application FastAPI
# =============================
app = FastAPI(
    title="API de Prédiction de Churn",
    description="API pour prédiction, mise à jour et réentraînement de modèle de churn client",
)
# 🔹 Variables globales
model = None
model_version = "1.0.0"  # Initial version of the model
EXPECTED_COLUMNS = []

# ✅ Ajout de l'initialisation de metrics
metrics = {"predictions": 0, "retrainings": 0, "errors": 0, "last_retraining": None}

# =============================
# 🔹 Chargement du modèle
# =============================

try:
    model = load_model()
    print("✅ Modèle chargé avec succès.")
    # 🔹 Ajout de la version du modèle
    model_version = "1.0.0"  # Version initiale du modèle
    print(f"✅ Modèle chargé avec succès. Version : {model_version}")

    # 🔹 Récupération de l'ordre exact attendu par le modèle
    EXPECTED_COLUMNS = list(model.feature_names_in_)
    print("📝 Colonnes attendues par le modèle :", EXPECTED_COLUMNS)

    # 🔹 Encodage des États (mapping si connu, sinon valeur par défaut)
    STATE_ENCODER = {
        "NY": 1,
        "CA": 2,
        "TX": 3,
        "NJ": 4,
        "WA": 5,
        # Ajouter les autres États ici
    }

except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None


# =============================
# 🔹 Schéma d'entrée pour la prédiction
# =============================
class InputData(BaseModel):
    state: str
    account_length: float
    area_code: float
    international_plan: int
    voice_mail_plan: int
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: float
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: float
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: float
    total_intl_charge: float
    customer_service_calls: float


# =============================
# 🔹 Mapping entre les colonnes API et celles du modèle
# =============================
COLUMN_MAPPING = {
    "state": "State",
    "account_length": "Account length",
    "area_code": "Area code",
    "international_plan": "International plan",
    "voice_mail_plan": "Voice mail plan",
    "number_vmail_messages": "Number vmail messages",
    "total_day_minutes": "Total day minutes",
    "total_day_calls": "Total day calls",
    "total_day_charge": "Total day charge",
    "total_eve_minutes": "Total eve minutes",
    "total_eve_calls": "Total eve calls",
    "total_eve_charge": "Total eve charge",
    "total_night_minutes": "Total night minutes",
    "total_night_calls": "Total night calls",
    "total_night_charge": "Total night charge",
    "total_intl_minutes": "Total intl minutes",
    "total_intl_calls": "Total intl calls",
    "total_intl_charge": "Total intl charge",
    "customer_service_calls": "Customer service calls",
}


# =============================
# 🔹 Route d'accueil
# =============================
@app.get("/", tags=["Home"])
def read_root():
    return {
        "message": "Bienvenue sur l'API de Prédiction de Churn!",
        "documentation_url": "/docs",
        "api_version": "1.0.0",
    }


# =============================
# 🔹 Route de prédiction
# =============================
@app.post(
    "/predict",
    tags=["Prediction"],
    description="Cette route effectue une prédiction du churn client en fonction des données fournies.",
)
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")

    try:
        # 🔄 Transformation en DataFrame
        df = pd.DataFrame([data.dict()])

        print("📌 DataFrame initial:")
        print(df)

        # 🔄 Renommer les colonnes pour correspondre au modèle
        df.rename(columns=COLUMN_MAPPING, inplace=True)

        # 🔄 Encodage de l'État (State)
        if df["State"][0] in STATE_ENCODER:
            df["State"] = STATE_ENCODER[df["State"][0]]
        else:
            df["State"] = 0  # Valeur par défaut si l'état n'est pas connu

        print("🔄 DataFrame après encodage de l'État:")
        print(df)

        # 🔄 **Replacer les colonnes dans le bon ordre**
        df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        print("🔄 DataFrame après mapping et réorganisation :")
        print(df)

        # 🔄 Prédiction
        prediction = model.predict(df)
        # ✅ **Mise à jour des métriques**
        metrics["predictions"] += 1
        print(f"✅ Prédiction réalisée : {prediction}")

        # 🔄 Résultat de la prédiction
        print(f"✅ Prédiction réalisée : {prediction}")
        return {"prediction": int(prediction[0])}

    except Exception as e:
        # ❌ **Enregistrer l'erreur dans les métriques**
        metrics["errors"] += 1
        print(f"❌ Erreur pendant la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 🔹 Route de Réentraînement du modèle
# =============================
@app.post(
    "/retrain",
    tags=["Model Retraining"],
    description="Réentraîne le modèle avec de nouvelles données d'entraînement.",
)
def retrain():
    try:
        print("🔄 Réentraînement du modèle...")

        # 🔄 Charger les données et préparer
        X_train, X_test, y_train, y_test = prepare_data("data/churn-bigml-80.csv")

        # 🔄 Réentraîner le modèle
        new_model = train_model(X_train, y_train)

        # 🔄 Sauvegarder le nouveau modèle
        save_model(new_model)

        # 🔄 Charger en mémoire
        global model, EXPECTED_COLUMNS
        model = new_model
        EXPECTED_COLUMNS = list(model.feature_names_in_)

        print("✅ Modèle réentraîné et chargé en mémoire.")
        return {"message": "Modèle réentraîné avec succès"}

    except Exception as e:
        print(f"❌ Erreur pendant le réentraînement : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 🔹 Détection de Drift (changement de distribution)
# =============================
app.get("/drift-detection", tags=["Drift Detection"])


def drift_detection():
    try:
        print("🔎 Chargement des données pour le drift detection...")
        X_train, _, _, _ = prepare_data("data/churn-bigml-80.csv")
        X_prod = pd.read_csv("data/new_data.csv")  # Données réelles

        print("📝 Colonnes dans le modèle :", list(X_train.columns))
        print("📝 Colonnes dans les données de production :", list(X_prod.columns))

        # Vérification des colonnes
        if set(X_train.columns) != set(X_prod.columns):
            raise ValueError(
                "Les colonnes entre le dataset d'entraînement et celui de production ne correspondent pas."
            )

        # 🔹 Détection de Drift
        drifts = {}
        for column in X_train.columns:
            stat, p_value = ks_2samp(X_train[column], X_prod[column])
            drifts[column] = {"statistic": stat, "p-value": p_value}

        print("✅ Drift Detection terminé.")
        return {"drift_detection": drifts}

    except Exception as e:
        print(f"❌ Erreur pendant la détection de drift : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 🔹 Importance des caractéristiques
# =============================
@app.get("/feature-importance", tags=["Features importance"])
def feature_importance():
    try:
        importance = model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": EXPECTED_COLUMNS, "importance": importance}
        )
        importance_df = importance_df.sort_values(by="importance", ascending=False)
        return {"feature_importance": importance_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 🔹 Vérification de l'état du modèle
# =============================
@app.get("/model-health", tags=["Model health"])
def model_health():
    try:
        print("🩺 Vérification de la santé du modèle...")
        health_score = (metrics["predictions"] - metrics["errors"]) / (
            metrics["predictions"] + 1
        )
        response = {
            "model_version": model_version,
            "metrics": metrics,
            "health_score": health_score,
        }
        print("✅ État du modèle :", response)
        return response

    except Exception as e:
        print(f"❌ Erreur pendant l'évaluation de la santé du modèle : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# 🔹 Exporter les métriques
# =============================
@app.get(
    "/export-metrics",
    tags=["Metrics"],
    summary="Export metrics as CSV",
    description="Exports the current metrics into a CSV file.",
)
def export_metrics():
    try:
        # 🔹 Nom du fichier CSV
        filename = "metrics_export.csv"

        # 🔹 Écriture dans le fichier
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            # 🔹 En-têtes
            writer.writerow(["Metric", "Value"])
            # 🔹 Valeurs
            for key, value in metrics.items():
                writer.writerow([key, value])

        print(f"✅ Les métriques ont été exportées dans le fichier '{filename}'")
        return FileResponse(filename, media_type="text/csv", filename=filename)

    except Exception as e:
        print(f"❌ Erreur lors de l'export des métriques : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de l'export des métriques : {str(e)}"
        )


# 🔄 Endpoint pour récupérer l'image
@app.get("/heatmap", tags=["heatMap"])
def get_heatmap():
    try:
        # 🔄 Chemin absolu de l'image
        file_path = os.path.join(os.getcwd(), "static/images/correlation_heatmap.png")

        if os.path.exists(file_path):
            return FileResponse(
                file_path, media_type="image/png", filename="correlation_heatmap.png"
            )
        else:
            raise FileNotFoundError("Image non trouvée")

    except Exception as e:
        print(f"❌ Erreur pendant le chargement de l'image : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant le chargement de l'image : {str(e)}",
        )


# =============================
# 🔹 Réentraînement programmé
# =============================
def scheduled_retraining():
    print("⏳ Scheduled retraining started...")
    X_train, _, y_train, _ = prepare_data("data/churn-bigml-80.csv")
    new_model = train_model(X_train, y_train)
    save_model(new_model)

    global model, model_version
    model = new_model
    model_version = str(float(model_version) + 0.1)  # Mise à jour de la version
    print(f"✅ Model retrained successfully. New version: {model_version}")


# Démarrage de l'auto-réentraînement chaque semaine
threading.Timer(604800, scheduled_retraining).start()


@app.get(
    "/model-version",
    tags=["Model"],
    summary="Get Model Version",
    description="Returns the current version of the loaded model.",
)
def get_model_version():
    try:
        return {"model_version": model_version}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de la version : {str(e)}",
        )


@app.get("/model-report")
def get_model_report():
    """
    Retourne le rapport HTML détaillé du modèle.
    """
    try:
        return FileResponse("model_report.html", media_type="text/html")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération du rapport : {str(e)}",
        )
