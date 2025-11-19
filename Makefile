# =======================
# VARIABLES
# =======================
VENV=venv
PYTHON=python3
PIP=$(VENV)/bin/pip
MAIN=main.py
REQUIREMENTS=requirements.txt

# =======================
# ENVIRONMENT CHECK
# =======================
.PHONY: check-env
check-env:
	@echo "🛡️  Vérification de l'environnement virtuel..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "⚠️  Le virtualenv n'existe pas. Création..."; \
		python3 -m venv $(VENV); \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r $(REQUIREMENTS); \
	else \
		echo "✅ Virtualenv détecté : $(VENV)"; \
	fi

# =======================
# INSTALLATION ET PRÉPARATION
# =======================
install: check-env
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

prepare: check-env
	$(PYTHON) $(MAIN) --prepare

run-mlflow:
	mlflow ui --host 0.0.0.0 --port 5000 &

train:
	make run-mlflow
	$(PYTHON) $(MAIN) --train

evaluate: check-env
	$(PYTHON) $(MAIN) --evaluate

save: check-env
	$(PYTHON) $(MAIN) --save

load: check-env
	$(PYTHON) $(MAIN) --load

# =======================
# OUTILS DE DÉVELOPPEMENT
# =======================
clean:
	@echo "🧹 Nettoyage des fichiers temporaires..."
	rm -rf __pycache__ *.joblib $(VENV) *.pkl *.html
	find . -type d -name "__pycache__" -exec rm -r {} + || true

lint: check-env
	flake8 --max-line-length=120 *.py

format: check-env
	black *.py

test: check-env
	PYTHONPATH=. pytest tests/ --maxfail=1 --disable-warnings

# =======================
# SERVEUR ET MONITORING
# =======================
serve: check-env
	uvicorn app:app --reload --host 0.0.0.0 --port 8084

watch: check-env
	venv/bin/watchmedo auto-restart --directory=./ --pattern="*.py" --recursive -- $(PYTHON) app.py

# =======================
# RAPPORT ET ANALYSES
# =======================
report: check-env
	@echo "📝 Génération du rapport..."
	$(PYTHON) $(MAIN) --evaluate

# =======================
# COMMANDE GLOBALE
# =======================
all: install prepare train evaluate save load
	@echo "🚀 Toutes les étapes ont été exécutées avec succès !"

# =======================
# SÉCURITÉ ET SCANNING
# =======================

# Scan de sécurité avec Safety
safety:
	@echo "🔍 Scan de sécurité avec Safety..."
	safety scan
	@echo "✅ Scan Safety terminé."

# Analyse de sécurité avec Bandit
secure:
	@echo "🔍 Analyse de sécurité avec Bandit..."
	bandit -r . -f html -o bandit_report.html
	@echo "✅ Rapport généré : bandit_report.html"
