FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulate.py .

# /data is a shared volume with the dashboard container
VOLUME ["/data"]

CMD ["python", "simulate.py"]
