FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY . .

RUN python generate_dummy_data.py \
    && python generate_mock_nr_artifact.py \
    && python -m experiments.synthetic_experiment --scenario default_run --seed 20260706 \
    && python -m pipeline.run

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
