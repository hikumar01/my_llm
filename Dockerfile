FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY frontend /app/frontend
COPY src /app/src

EXPOSE 8080
CMD ["python", "src/api_server.py"]
