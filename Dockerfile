FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# Include application code for isolated Nextflow processes
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Allow Python to import project modules from any working directory
ENV PYTHONPATH=/app
