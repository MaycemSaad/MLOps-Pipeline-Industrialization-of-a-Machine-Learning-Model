# model_pipeline.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn
import psutil
import datetime
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
import logging
from mlflow import log_metric, log_param
from sklearn.metrics import classification_report, accuracy_score
import base64


def collect_system_metrics():
    """Collect system metrics for logging."""
    metrics = {}
    metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    metrics["memory_percent"] = memory_info.percent

    if psutil.sensors_temperatures():
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            for entry in entries:
                metrics[f"{name}_{entry.label}_temp"] = entry.current
    return metrics


def log_to_elasticsearch(metrics: dict, model_name: str, dataset: str):
    """Log metrics to Elasticsearch."""
    system_metrics = collect_system_metrics()

    # Log model metrics
    for key, value in metrics.items():
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metric": key,
            "value": value,
            "model": model_name,
            "dataset": dataset,
        }
        try:
            es.index(index="mlflow-metrics", body=log_entry)
        except exceptions.ConnectionError as e:
            print(f"Connection error to Elasticsearch: {e}")
        except exceptions.RequestError as e:
            print(f"Request error in Elasticsearch: {e}")

    # Log system metrics
    for key, value in system_metrics.items():
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metric": key,
            "value": value,
            "model": model_name,
            "dataset": dataset,
            "system_metric": True,
        }
        try:
            es.index(index="mlflow-metrics", body=log_entry)
        except exceptions.ConnectionError as e:
            print(f"Connection error to Elasticsearch: {e}")
        except exceptions.RequestError as e:
            print(f"Request error in Elasticsearch: {e}")


def prepare_data(path):
    """
    Charger et prétraiter les données :
    - Lecture du CSV
    - Encodage des variables catégorielles
    - Affichage info, description et heatmap
    - Séparation X / y + train_test_split
    """
    print("📥 Chargement des données...")
    data = pd.read_csv(path)
    print("✅ Données chargées.")
    print("\n📝 Vérification des valeurs nulles :")
    print(data.isnull().sum())
    print("\n📝 Vérification des types de données :")
    print(data.dtypes)

    print("\n📝 Vérification des doublons :")
    print(data.duplicated().sum())

    print("\n📊 Aperçu des données :")
    print(data.head())

    print("\n📏 Dimensions du dataset :", data.shape)

    print("\nℹ️ Informations générales :")
    print(data.info())

    print("\n📈 Statistiques descriptives :")
    print(data.describe())

    print("\n🛠️ Vérification après préparation :")
    print(data.head())
    print(f"Dimensions après préparation : {data.shape}")

    # 🔄 Vérifier si le dossier existe, sinon le créer
    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    # 🔄 Affichage de la heatmap et sauvegarde dans le bon dossier
    print("\n🔥 Heatmap des corrélations :")
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        data.select_dtypes(include=["float64", "int64"]).corr(),
        annot=True,
        cmap="coolwarm",
    )
    plt.title("Heatmap avant encodage")
    plt.savefig("static/images/correlation_heatmap.png")  # 🔹 Nouveau chemin
    plt.close()
    print("✅ Heatmap sauvegardée sous 'static/images/correlation_heatmap.png'")

    # Encodage des variables catégorielles
    print("\n🔠 Encodage des variables catégorielles...")
    encoder = LabelEncoder()
    for col in data.select_dtypes(include="object").columns:
        data[col] = encoder.fit_transform(data[col])
    print("✅ Encodage terminé.")

    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


def log_visualization_to_elasticsearch(image_path, model_name, dataset):
    """Log image visualization to Elasticsearch."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model_name,
            "dataset": dataset,
            "image": encoded_string,
            "image_type": image_path.split(".")[-1],
        }
        try:
            es.index(index="mlflow-visualizations", body=log_entry)
        except exceptions.ConnectionError as e:
            print(f"Connection error to Elasticsearch: {e}")
        except exceptions.RequestError as e:
            print(f"Request error in Elasticsearch: {e}")


def log_system_metrics():
    """Log system metrics to MLFlow during training."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Utilisation de noms de métriques simples
    mlflow.log_metric("CPU_Usage_Percentage", cpu_percent)  # Remplacement effectué ici
    mlflow.log_metric("Memory_Usage_Percentage", memory_info.percent)  # Et ici

    # Vérification des températures et enregistrement des données
    if psutil.sensors_temperatures():
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            for entry in entries:
                # Remplacer les espaces dans les noms des métriques
                metric_name = f"{name}_{entry.label}".replace(" ", "_")
                mlflow.log_metric(metric_name, entry.current)


def plot_confusion_matrix(y_true, y_pred, labels):
    """Generate and log confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_pred):
    """Generate and log ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()


def train_model(X_train, y_train):
    print("💡 Entraînement du modèle...")

    # 📌 Démarrage de l'expérience MLFlow
    with mlflow.start_run():
        start_time = time.time()

        # 📌 Log des paramètres
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # 📌 Entraînement
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)

        # 📌 Log des métriques
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", accuracy)

        # 🔍 Prédictions sur le set de train pour les métriques
        y_pred = model.predict(X_train)

        # 📌 Log des métriques avancées
        report = classification_report(y_train, y_pred, output_dict=True)
        mlflow.log_metric("Precision", report["weighted avg"]["precision"])
        mlflow.log_metric("Recall", report["weighted avg"]["recall"])
        mlflow.log_metric("F1_Score", report["weighted avg"]["f1-score"])

        # ⏲️ Log du temps d'entraînement (le nom est modifié pour être accepté par MLflow)
        end_time = time.time()
        mlflow.log_metric("Training_Time_seconds", end_time - start_time)

        # 🔍 Log des métriques système

        # 📊 Génération des artefacts
        plot_confusion_matrix(y_train, y_pred, labels=model.classes_)
        plot_roc_curve(y_train, y_pred)

        # 💾 Enregistrement du modèle
        mlflow.sklearn.log_model(model, "model")
        print("✅ Modèle enregistré avec MLFlow.")
        # 📌 Log du modèle versionné dans le registre MLflow
        client = MlflowClient()
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        # Ajouter à un modèle registry
        model_name = "Churn_Prediction_Model"

        try:
            # Enregistrer la version du modèle
            version = client.create_model_version(
                model_name, model_uri, mlflow.active_run().info.run_id
            )

            # Ajouter des informations supplémentaires (description)
            client.update_model_version(
                model_name,
                version.version,
                description="Churn prediction model using Random Forest",
            )

            # Ajouter des tags à la version du modèle
            client.set_model_version_tag(
                model_name, version.version, "model_type", "RandomForest"
            )
            client.set_model_version_tag(
                model_name, version.version, "domain", "Churn Prediction"
            )
            print(f"✅ Modèle versionné et enregistré : Version {version.version}")
        except Exception as e:
            print(f"⚠️ Erreur lors de l'enregistrement du modèle versionné : {e}")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle à l'aide de la précision et d'un rapport de classification.
    """
    print("🧪 Évaluation du modèle...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\n🎯 Accuracy: {acc:.4f}")
    print("\n📋 Rapport de classification :\n")
    print(classification_report(y_test, predictions))
    # Dans la fonction `evaluate_model`:
    generate_html_report(model, X_test, y_test)
    # Generate classification report and store it
    report = classification_report(y_test, predictions, output_dict=True)
    print(classification_report(y_test, predictions))

    # Prepare metrics for Elasticsearch
    # Prepare metrics for Elasticsearch
    metrics = {
        "test_accuracy": acc,
        "test_precision": report["weighted avg"]["precision"],
        "test_recall": report["weighted avg"]["recall"],
        "test_f1_score": report["weighted avg"]["f1-score"],
        "roc_auc": auc(*roc_curve(y_test, model.predict_proba(X_test)[:, 1])[:2]),
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Churn", "Churn"],
        yticklabels=["Non-Churn", "Churn"],
    )
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    print("✅ Visualisations sauvegardées : confusion_matrix.png, roc_curve.png")


def save_model(model, filename="model.joblib"):
    """
    Sauvegarde le modèle entraîné avec joblib.
    """
    joblib.dump(model, filename)
    print(f"💾 Modèle sauvegardé sous '{filename}'.")


def load_model(filename="model.joblib"):
    """
    Charge un modèle sauvegardé avec joblib.
    """
    print(f"📂 Chargement du modèle depuis '{filename}'...")
    return joblib.load(filename)


def save_data(X_train, X_test, y_train, y_test, filename="data_split.pkl"):
    """
    Sauvegarde les datasets d'entraînement et de test.
    """
    with open(filename, "wb") as f:
        joblib.dump((X_train, X_test, y_train, y_test), f)
    print(f"💾 Données sauvegardées sous '{filename}'.")


def load_data(filename="data_split.pkl"):
    """
    Charge les datasets d'entraînement et de test.
    """
    print(f"📂 Chargement des données depuis '{filename}'...")
    with open(filename, "rb") as f:
        return joblib.load(f)


def generate_html_report(model, X_test, y_test, filename="model_report.html"):
    """
    Génère un rapport HTML détaillé avec les métriques, les courbes ROC,
    la matrice de confusion et l'importance des features.
    """
    print("📊 Génération du rapport HTML...")

    # Prédictions
    y_pred = model.predict(X_test)

    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Feature Importances
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"Feature": X_test.columns, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)
    else:
        feature_importance = pd.DataFrame()

    # Courbe ROC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Création des graphiques
    if not os.path.exists("reports"):
        os.makedirs("reports")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion")
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC")
    plt.legend()
    plt.savefig("reports/roc_curve.png")
    plt.close()

    # Génération du HTML
    with open(filename, "w") as file:
        file.write(f"<h1>Rapport de Modèle</h1>")
        file.write("<h2>Métriques de Classification</h2>")
        file.write(report_df.to_html())

        file.write("<h2>Importance des Features</h2>")
        if not feature_importance.empty:
            file.write(feature_importance.to_html())
        else:
            file.write("<p>Le modèle n'a pas d'importance de features.</p>")

        file.write("<h2>Matrice de Confusion</h2>")
        file.write('<img src="reports/confusion_matrix.png" width="600">')

        file.write("<h2>Courbe ROC</h2>")
        file.write('<img src="reports/roc_curve.png" width="600">')

    print(f"📌 Rapport généré avec succès : {filename}")


def log_system_metrics():
    """Log system metrics to MLFlow during training."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    mlflow.log_metric("CPU Usage (%)", cpu_percent)
    mlflow.log_metric("Memory Usage (%)", memory_info.percent)

    if psutil.sensors_temperatures():
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            for entry in entries:
                mlflow.log_metric(f"{name}_{entry.label}", entry.current)
