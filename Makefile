.PHONY: help install train evaluate test lint format run-api run-ui docker-build docker-up docker-down

help:
	@echo "Customer Churn Prediction — Developer Makefile"
	@echo "--------------------------------------------------"
	@echo "make install      : Install production and dev dependencies"
	@echo "make train        : Run 5-fold CV training pipeline and serialize model"
	@echo "make evaluate     : Run financial threshold optimization & produce charts"
	@echo "make test         : Run automated Pytest unit and integration test suite"
	@echo "make lint         : Run Ruff linter"
	@echo "make format       : Auto-fix linting issues with Ruff"
	@echo "make run-api      : Start FastAPI REST service with live reload (port 8000)"
	@echo "make run-ui       : Start Streamlit interactive dashboard (port 8501)"
	@echo "make docker-build : Build container images with Docker Compose"
	@echo "make docker-up    : Launch FastAPI and Streamlit services via Docker Compose"
	@echo "make docker-down  : Stop all running Docker Compose containers"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/train.py

evaluate:
	python src/evaluate.py

test:
	pytest -v

lint:
	ruff check .

format:
	ruff check --fix .

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run app/app.py --server.port 8501

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
