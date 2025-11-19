# Utilisation d'une image Python de base
FROM python:3.9-slim

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers du projet dans le conteneur
COPY . /app/

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel FastAPI va tourner
EXPOSE 8080

# Commande pour lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
