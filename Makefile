.PHONY: help env install lint fmt test clean train serve docs

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
env: ## Create and activate virtual environment
	poetry install --with dev

install: ## Install dependencies
	poetry install

# Code quality
lint: ## Run linting checks
	ruff check .
	black --check .
	isort --check-only .
	mypy src/

fmt: ## Format code
	black .
	isort .
	ruff check --fix .

# Testing
test: ## Run tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v

test-cov: ## Run tests with coverage
	pytest --cov=src/bloodwork_ai --cov-report=html --cov-report=term-missing

# Data and model operations
data-prepare: ## Prepare sample data
	python scripts/generate_sample_data.py

dvc-repro: ## Run DVC pipeline
	dvc repro

dvc-push: ## Push data to remote
	dvc push

dvc-pull: ## Pull data from remote
	dvc pull

# Training
train-anemia: ## Train anemia classifier
	python -m src.bloodwork_ai.training.train_tabular --task cbc.anemia

train-mets: ## Train metabolic syndrome predictor
	python -m src.bloodwork_ai.training.train_tabular --task cmp.mets

train-cvd: ## Train CVD risk predictor
	python -m src.bloodwork_ai.training.train_tabular --task lipid.cvd

train-thyroid: ## Train thyroid dysfunction classifier
	python -m src.bloodwork_ai.training.train_tabular --task thyroid.dysfunction

train-dic: ## Train DIC early warning system
	python -m src.bloodwork_ai.training.train_tabular --task coag.dic

train-b12: ## Train B12 deficiency predictor
	python -m src.bloodwork_ai.training.train_tabular --task vitamins.b12

train-ferritin: ## Train ferritin low predictor
	python -m src.bloodwork_ai.training.train_tabular --task iron.ferritin

train-wbc: ## Train WBC detection model
	python -m src.bloodwork_ai.training.train_vision --config configs/models/vision/yolo_wbc.yaml

train-rbc: ## Train RBC detection model
	python -m src.bloodwork_ai.training.train_vision --config configs/models/vision/yolo_rbc.yaml

train-platelet: ## Train platelet detection model
	python -m src.bloodwork_ai.training.train_vision --config configs/models/vision/yolo_platelet.yaml

train-all: ## Train all models
	$(MAKE) train-anemia train-mets train-cvd train-thyroid train-dic train-b12 train-ferritin train-wbc train-rbc train-platelet

# Evaluation
eval: ## Run evaluation pipeline
	python -m src.bloodwork_ai.evaluation.report_builder

eval-anemia: ## Evaluate anemia model
	python -m src.bloodwork_ai.evaluation.eval_tabular --task cbc.anemia

eval-wbc: ## Evaluate WBC model
	python -m src.bloodwork_ai.evaluation.eval_vision --model yolo_wbc

# Serving
serve: ## Start FastAPI server
	uvicorn src.bloodwork_ai.serving.fastapi_app:app --host 0.0.0.0 --port 8080 --reload

serve-prod: ## Start production server
	uvicorn src.bloodwork_ai.serving.fastapi_app:app --host 0.0.0.0 --port 8080 --workers 4

# MLflow
mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-server: ## Start MLflow server
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Docker
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile.training -t bloodwork-ai:training .
	docker build -f docker/Dockerfile.serving -t bloodwork-ai:serving .

docker-run: ## Run Docker container
	docker-compose -f docker/compose.yaml up

# Kubernetes
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f k8s/

# Documentation
docs: ## Generate documentation
	mkdir -p docs
	python scripts/generate_docs.py

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

clean-data: ## Clean up data files
	rm -rf data/interim/
	rm -rf data/processed/

clean-models: ## Clean up model artifacts
	rm -rf artifacts/

clean-all: clean clean-data clean-models ## Clean everything

# Development
dev-setup: env data-prepare ## Complete development setup
	@echo "Development environment ready!"
	@echo "Run 'make serve' to start the API server"
	@echo "Run 'make mlflow-ui' to start MLflow UI"

# CI/CD
ci: lint test ## Run CI pipeline
	@echo "CI pipeline completed successfully"

# Full pipeline
pipeline: clean data-prepare dvc-repro eval ## Run full pipeline
	@echo "Full pipeline completed successfully"
