# One small image for the scheduler, the workers and the dashboard (no torch: the PoC embeds with
# Part 1's model-free HashEmbedder). Build context is the repository root, see docker-compose.yml.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "numpy==2.2.*" "psycopg[binary]==3.2.*"

COPY patsearch/ /app/patsearch/
COPY poc/pipeline/ /app/pipeline/

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
