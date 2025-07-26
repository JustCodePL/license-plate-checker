# System Rozpoznawania Tablic Rejestracyjnych

Aplikacja do rozpoznawania tablic rejestracyjnych z obrazu w czasie rzeczywistym, oparta o PaddleOCR, z obsługą GPU oraz możliwością wysyłania wyników do webhooka.

## Wymagania
- Python 3.8+
- Kamera USB lub IP
- (Opcjonalnie) Karta graficzna NVIDIA z obsługą CUDA do przyspieszenia na GPU

## Instalacja

### Opcja 1: Instalacja lokalna

1. **Klonuj repozytorium:**
   ```bash
   git clone <adres_repozytorium>
   cd tablice
   ```

2. **Zainstaluj wymagane pakiety:**
   - Dla CPU:
     ```bash
     pip install -r requirements.txt
     pip install paddleocr paddlepaddle
     ```
   - Dla GPU (zalecane, jeśli masz kartę NVIDIA):
     ```bash
     pip install -r requirements.txt
     pip install paddleocr paddlepaddle-gpu
     ```
     > **Uwaga:**
     > Jeśli chcesz konkretną wersję CUDA, sprawdź: https://www.paddlepaddle.org.cn/install/quick

3. **(Opcjonalnie) Zainstaluj sterowniki kamery oraz OpenCV, jeśli nie są obecne:**
   ```bash
   pip install opencv-python
   ```

### Opcja 2: Uruchomienie w Docker

#### Wymagania Docker:
- Docker i Docker Compose
- (Dla GPU) NVIDIA Docker Runtime

#### Szybkie uruchomienie z Docker Hub:

1. **Pobierz i uruchom obraz z Docker Hub:**
   ```bash
   docker run --gpus all -e CAMERA_URL=http://twoja.kamera.ip:port/video justcodepl/license-plate-checker
   ```

2. **Uruchom z Docker Compose (jeśli masz docker-compose.yml):**
   ```bash
   docker-compose up
   ```

#### Budowanie lokalne:

1. **Buduj i uruchom z Docker Compose:**
   ```bash
   docker-compose up --build
   ```
   > **Uwaga:** Pierwsze budowanie może potrwać dłużej - modele PaddleOCR są pobierane podczas budowania obrazu.

2. **Lub uruchom tylko główną aplikację:**
   ```bash
   docker-compose up license-plate-detector
   ```

3. **Uruchom z symulatorem kamery (do testów):**
   ```bash
   docker-compose --profile camera-simulator up
   ```

#### Uruchomienie bez Docker Compose:
```bash
# Buduj obraz (pobierze modele podczas budowania)
docker build -t license-plate-detector .

# Uruchom kontener
docker run --gpus all -e CAMERA_URL=http://twoja.kamera.ip:port/video license-plate-detector
```

## Konfiguracja

Aplikacja korzysta z kilku zmiennych środowiskowych:

- `CAMERA_URL` – adres URL kamery IP lub numer (np. `0` dla USB, np. `http://192.168.1.1:8080/video` dla kamery IP)
- `DEBUG_MODE` – tryb debugowania: `true` lub `false` (domyślnie: `true`)
- `WEBHOOK_URL` – adres webhooka do wysyłania wykrytych tablic (domyślnie: pusty)

Możesz ustawić je w systemie lub utworzyć plik `.env`.

Przykład uruchomienia z parametrami:
```bash
CAMERA_TYPE=usb CAMERA_URL=0 DEBUG_MODE=false WEBHOOK_URL=https://twoj.webhook.url python main.py
```

### Konfiguracja Docker:
Edytuj `docker-compose.yml` lub przekaż zmienne środowiskowe:
```bash
# Uruchomienie z Docker Hub
docker run --gpus all -e CAMERA_URL=http://192.168.1.100:8080/video -e WEBHOOK_URL=https://twoj.webhook.url justcodepl/license-plate-checker

# Lub dla lokalnie zbudowanego obrazu
docker run -e CAMERA_URL=http://192.168.1.100:8080/video -e WEBHOOK_URL=https://twoj.webhook.url license-plate-detector
```

## Uruchomienie

### Lokalnie:
```bash
python main.py
```

### W Docker:
```bash
docker-compose up
```

Pierwsze uruchomienie może potrwać dłużej (pobieranie modeli OCR).

## Działanie
- Program wykrywa tablice rejestracyjne z obrazu kamery.
- Wykryte tablice są wypisywane w konsoli oraz wysyłane do webhooka (jeśli skonfigurowano `WEBHOOK_URL`).
- Obsługa GPU jest automatyczna, jeśli zainstalowano `paddlepaddle-gpu` i wykryto kartę graficzną.

## Najczęstsze problemy
- **Brak modułu:** Zainstaluj brakujące pakiety poleceniem `pip install ...`.
- **Brak obrazu z kamery:** Sprawdź poprawność `CAMERA_URL` i podłączenie kamery.
- **Brak wsparcia GPU:** Upewnij się, że masz zainstalowane `paddlepaddle-gpu` zgodne z Twoją wersją CUDA.
- **Problem z Docker GPU:** Sprawdź czy masz zainstalowany NVIDIA Docker Runtime.

## Kontakt
W razie problemów lub pytań: Artur Czuba