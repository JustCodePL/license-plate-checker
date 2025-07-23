# System Rozpoznawania Tablic Rejestracyjnych

Aplikacja do rozpoznawania tablic rejestracyjnych z obrazu w czasie rzeczywistym, oparta o PaddleOCR, z obsługą GPU oraz możliwością wysyłania wyników do webhooka.

## Wymagania
- Python 3.8+
- Kamera USB lub IP
- (Opcjonalnie) Karta graficzna NVIDIA z obsługą CUDA do przyspieszenia na GPU

## Instalacja

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

## Uruchomienie

```bash
python main.py
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

## Kontakt
W razie problemów lub pytań: Artur Czuba