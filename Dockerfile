# Użyj oficjalnego obrazu Python z CUDA dla obsługi GPU
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Ustaw zmienne środowiskowe
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Zainstaluj systemowe zależności
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Utwórz katalog roboczy
WORKDIR /app

# Skopiuj pliki wymagań
COPY requirements.txt .

# Zainstaluj zależności Python
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Zainstaluj PaddlePaddle z obsługą GPU
RUN pip3 install --no-cache-dir paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Zainstaluj PaddleOCR
RUN pip3 install --no-cache-dir paddleocr

# Skopiuj kod aplikacji
COPY main.py .

# Utwórz skrypt do pobrania modeli podczas budowania
RUN echo '#!/bin/bash\n\
echo "Pobieranie modeli PaddleOCR..."\n\
python3 -c "\n\
from paddleocr import PaddleOCR\n\
import os\n\
\n\
# Ustaw katalog cache na lokalny katalog w kontenerze\n\
os.environ[\"HOME\"] = \"/app\"\n\
\n\
# Inicjalizuj PaddleOCR aby pobrać modele\n\
ocr = PaddleOCR(\n\
    use_angle_cls=True,\n\
    lang=\"en\",\n\
    use_gpu=False,  # Podczas budowania używamy CPU\n\
    show_log=False,\n\
    drop_score=0.3\n\
)\n\
print(\"Modele PaddleOCR pobrane pomyślnie!\")\n\
"\n\
echo "Budowanie obrazu zakończone!"' > /app/download_models.sh

# Nadaj uprawnienia do wykonania skryptu
RUN chmod +x /app/download_models.sh

# Pobierz modele podczas budowania obrazu
RUN /app/download_models.sh

# Utwórz użytkownika nie-root dla bezpieczeństwa
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Ustaw domyślne zmienne środowiskowe
ENV CAMERA_TYPE=ip
ENV CAMERA_URL=http://192.168.1.124:8080/video
ENV DEBUG_MODE=true
ENV WEBHOOK_URL=""

# Eksponuj port (jeśli aplikacja będzie miała interfejs web)
EXPOSE 8000

# Uruchom aplikację
CMD ["python3", "main.py"]