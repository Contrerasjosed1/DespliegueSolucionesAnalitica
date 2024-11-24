FROM python:3.10-slim

# Instalar herramientas de compilación
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "api.py"]
