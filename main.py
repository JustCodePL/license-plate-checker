import os
import time
import cv2
import easyocr
import numpy as np
import sys
import threading

# Pobierz konfigurację z ENV
CAMERA_TYPE = os.getenv('CAMERA_TYPE', 'ip')
CAMERA_URL = os.getenv('CAMERA_URL', 'http://192.168.1.124:8080/video')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'  # Włącz debug domyślnie

# Inicjalizacja OCR z mniejszym modelem i timeout
ocr = None

def get_camera():
    if CAMERA_TYPE == 'usb':
        try:
            cam_index = int(CAMERA_URL)
        except ValueError:
            cam_index = 0
        cap = cv2.VideoCapture(cam_index)
    elif CAMERA_TYPE == 'ip':
        cap = cv2.VideoCapture(CAMERA_URL)
    else:
        raise ValueError('Nieznany typ kamery: {}'.format(CAMERA_TYPE))
    if not cap.isOpened():
        raise RuntimeError('Nie można otworzyć kamery!')
    return cap

def init_ocr():
    global ocr
    if ocr is None:
        try:
            print("Inicjalizacja EasyOCR z optymalizacją dla tablic rejestracyjnych...")
            print("Wyłączanie GPU dla stabilności...")
            print("Pobieranie modeli może potrwać kilka minut przy pierwszym uruchomieniu...")

            # EasyOCR z językami polskim i angielskim dla lepszego rozpoznawania polskich tablic
            # Na Windows używamy domyślnej lokalizacji dla modeli
            # gpu=False aby uniknąć problemów z CUDA warnings
            model_storage_dir = os.path.expanduser('~/.EasyOCR')

            ocr = easyocr.Reader(
                ['en'],  # Angielski model jest najlepszy dla tablic rejestracyjnych
                model_storage_directory=model_storage_dir,
                gpu=False,
                verbose=False,  # wyłącz verbose aby zmniejszyć warnings
                download_enabled=True  # pozwól na pobieranie modeli przy pierwszym uruchomieniu
            )

            print("EasyOCR gotowy z modelem angielskim (optymalny dla tablic)!")
            print("Kończę init_ocr()...")
        except Exception as e:
            print(f"Błąd inicjalizacji OCR: {e}")
            import traceback
            traceback.print_exc()
            ocr = None

def detect_license_plate_regions(frame):
    """Wykryj potencjalne regiony tablic rejestracyjnych"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie krawędzi
    edges = cv2.Canny(gray, 50, 150)

    # Znajdź kontury
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_plates = []

    for contour in contours:
        # Oblicz prostokąt otaczający
        x, y, w, h = cv2.boundingRect(contour)

        # Sprawdź proporcje (tablice mają charakterystyczne proporcje)
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Filtruj na podstawie proporcji i rozmiaru
        if (2.0 <= aspect_ratio <= 6.0 and  # Typowe proporcje tablic
            area >= 1000 and  # Minimalny rozmiar
            w >= 80 and h >= 20):  # Minimalne wymiary

            # Dodaj margines wokół wykrytego regionu
            margin = 10
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(frame.shape[1], x + w + margin)
            y_end = min(frame.shape[0], y + h + margin)

            roi = frame[y_start:y_end, x_start:x_end]
            if roi.size > 0:
                potential_plates.append(roi)

    return potential_plates if potential_plates else [frame]

def preprocess_for_ocr(frame):
    """Przetwarzanie obrazu dla lepszego rozpoznawania OCR"""
    # Konwersja do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Zwiększenie kontrastu przy użyciu CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Filtracja Gaussa dla redukcji szumu
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Morfologia - zamknięcie luk w tekście
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # Zwiększenie ostrości
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(processed, -1, kernel_sharp)

    return sharpened

def detect_and_read_plate(frame):
    global ocr
    # OCR powinien być już zainicjalizowany w main()
    if ocr is None:
        return None

    try:
        # Najpierw wykryj potencjalne regiony tablic
        potential_regions = detect_license_plate_regions(frame)

        if DEBUG_MODE:
            print(f"Wykryto {len(potential_regions)} potencjalnych regionów tablic")

        # Jeśli nie wykryto regionów, użyj całej klatki
        if not potential_regions or len(potential_regions) == 0:
            potential_regions = [frame]
            if DEBUG_MODE:
                print("Używam całej klatki do analizy")

        all_candidates = []

                # Przeanalizuj każdy region osobno
        for i, region in enumerate(potential_regions):
            if DEBUG_MODE:
                print(f"Analizuję region {i+1}/{len(potential_regions)}")

            # Przetwarzanie obrazu dla lepszego OCR
            processed_frame = preprocess_for_ocr(region)

            # OCR z uproszczonymi parametrami
            result = ocr.readtext(processed_frame)

            if result and len(result) > 0:
                if DEBUG_MODE:
                    print(f"Region {i+1} OCR wykrył {len(result)} tekstów:")

                # Przeszukaj wszystkie wykryte teksty w tym regionie
                for detection in result:
                    if len(detection) >= 2:  # Można mieć tylko 2 elementy zamiast 3
                        text = detection[1].strip() if len(detection) > 1 else ""
                        confidence = detection[2] if len(detection) > 2 else 1.0  # Domyślna pewność
                        bbox = detection[0] if len(detection) > 0 else None

                        if DEBUG_MODE:
                            print(f"  '{text}' (pewność: {confidence:.2f})")

                        # Filtruj potencjalne tablice rejestracyjne
                        if is_license_plate(text) and confidence > 0.1:  # Bardzo niski próg pewności
                            all_candidates.append((text, confidence, bbox))
                            if DEBUG_MODE:
                                print(f"  -> DODANO DO KANDYDATÓW")
                        elif DEBUG_MODE:
                            print(f"  -> ODRZUCONO: is_plate={is_license_plate(text)}, confidence={confidence:.2f}")
            elif DEBUG_MODE:
                print(f"Region {i+1}: Brak wykryć OCR")

                # Wybierz najlepszego kandydata ze wszystkich regionów
        if all_candidates:
            if DEBUG_MODE:
                print(f"Znaleziono {len(all_candidates)} kandydatów na tablice:")
                for text, conf, _ in all_candidates:
                    print(f"  - '{text}' ({conf:.2f})")

            # Sortuj według pewności
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate = all_candidates[0]

            if DEBUG_MODE:
                print(f"Najlepszy kandydat: '{best_candidate[0]}' ({best_candidate[1]:.2f})")

            # Dodatkowa walidacja najlepszego kandydata
            clean_text = clean_license_plate_text(best_candidate[0])
            if DEBUG_MODE:
                print(f"Po czyszczeniu: '{clean_text}'")

            if clean_text and len(clean_text) >= 3:  # Obniżony próg
                return clean_text
            elif DEBUG_MODE:
                print("Kandydat odrzucony po czyszczeniu")

        elif DEBUG_MODE:
            print("Brak kandydatów na tablice rejestracyjne")
    except Exception as e:
        print(f"Błąd OCR: {e}")
        return None
    return None

def clean_license_plate_text(text):
    """Czyści i normalizuje tekst tablicy rejestracyjnej"""
    import re

    # Usuń zbędne znaki i spacje
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Minimalnie popraw tylko oczywiste błędy OCR
    # Tylko gdy jesteśmy pewni, że to błąd OCR, nie rzeczywista litera
    replacements = {
        'O': '0',  # O -> 0 tylko gdy otoczone cyframi
        'I': '1',  # I -> 1 tylko gdy otoczone cyframi
    }

    # Bardzo konserwatywne zastępowanie - tylko oczywiste przypadki
    result = ""
    for i, char in enumerate(clean_text):
        # Zamień tylko gdy znak jest otoczony cyframi z obu stron
        if (char in replacements and char.isalpha() and
            i > 0 and i < len(clean_text) - 1 and
            clean_text[i-1].isdigit() and clean_text[i+1].isdigit()):
            result += replacements[char]
        else:
            result += char

    return result

def is_license_plate(text):
    """Sprawdź czy tekst może być tablicą rejestracyjną - optymalizowane dla polskich tablic"""
    import re

    # Usuń spacje i znaki specjalne
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Sprawdź długość (polskie tablice: 3-8 znaków - obniżony próg)
    if len(clean_text) < 3 or len(clean_text) > 8:
        return False

    # Sprawdź czy zawiera cyfry i litery (typowe dla tablic)
    has_letters = bool(re.search(r'[A-Z]', clean_text))
    has_numbers = bool(re.search(r'[0-9]', clean_text))

    if not (has_letters and has_numbers):
        return False

    # Dodatkowe wzorce dla polskich tablic rejestracyjnych
    polish_patterns = [
        r'^[A-Z]{2,3}[0-9]{2,5}$',      # Standard: XX123, XXX1234
        r'^[A-Z]{1,2}[0-9]{3,4}[A-Z]{1,2}$',  # Starsze: X123Y, XX12YZ
        r'^[0-9]{2,3}[A-Z]{2,3}[0-9]{2,3}$',  # Alternatywne: 12XX34
        r'^[A-Z][0-9]{4,5}$',           # Specjalne: X12345
        r'^[0-9][A-Z]{2}[0-9]{3,4}$'   # Inne: 1XX234
    ]

    # Sprawdź czy pasuje do któregoś z wzorców
    for pattern in polish_patterns:
        if re.match(pattern, clean_text):
            return True

    # Podstawowe sprawdzenie struktury (fallback)
    # Minimum 2 litery i 2 cyfry
    letter_count = len(re.findall(r'[A-Z]', clean_text))
    digit_count = len(re.findall(r'[0-9]', clean_text))

    if letter_count >= 2 and digit_count >= 2:
        return True

    # Odrzuć oczywiste błędy
    if clean_text in ["TEST", "ERROR", "NULL", "NONE", "VOID"]:
        return False

    return False

def main():
    # Inicjalizacja OCR przed połączeniem z kamerą
    print("Inicjalizacja systemu rozpoznawania tablic...")
    init_ocr()
    print("Po wywołaniu init_ocr()")

    if ocr is None:
        print("Błąd: Nie można zainicjalizować OCR!")
        return

    print("OCR zainicjalizowany, łączę z kamerą...")

    # Połączenie z kamerą
    try:
        cap = get_camera()
        print(f"Połączono z kamerą ({CAMERA_TYPE}): {CAMERA_URL}")
    except Exception as e:
        print(f"Błąd połączenia z kamerą: {e}")
        return

    print("System gotowy - rozpoczynam rozpoznawanie tablic rejestracyjnych...")

    try:
        frame_count = 0
        last_detection_time = 0
        detection_cooldown = 2  # Sekundy między detekcjami dla tej samej tablicy
        last_detected_plate = ""
        consecutive_detections = {}  # Zliczanie kolejnych wykryć tej samej tablicy

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Brak obrazu z kamery!")
                if DEBUG_MODE:
                    print(f"Kamera: {CAMERA_TYPE}, URL: {CAMERA_URL}")
                time.sleep(1)
                continue

            if DEBUG_MODE and frame_count == 0:
                print(f"Pierwsza klatka otrzymana! Rozmiar: {frame.shape}")

            frame_count += 1
            current_time = time.time()

            # Przetwarzaj co 3 klatki dla lepszej wydajności
            if frame_count % 3 != 0:
                continue

            if DEBUG_MODE and frame_count % 30 == 0:
                print(f"Przetworzono {frame_count} klatek...")

            if DEBUG_MODE:
                print(f"\n--- ANALIZA KLATKI {frame_count} ---")

            # Wykrywanie tablicy
            plate = detect_and_read_plate(frame)

            if DEBUG_MODE:
                print(f"Wynik analizy: {plate if plate else 'BRAK'}")
                print("--- KONIEC ANALIZY ---\n")

            if plate:
                # Sprawdź czy to ta sama tablica co poprzednio
                if plate == last_detected_plate and (current_time - last_detection_time) < detection_cooldown:
                    continue

                # Zliczaj kolejne wykrycia tej samej tablicy dla potwierdzenia
                if plate in consecutive_detections:
                    consecutive_detections[plate] += 1
                else:
                    consecutive_detections[plate] = 1
                    # Wyczyść stare wykrycia
                    for old_plate in list(consecutive_detections.keys()):
                        if old_plate != plate:
                            consecutive_detections[old_plate] = 0

                # Wyświetl tylko jeśli tablica została wykryta co najmniej 2 razy
                if consecutive_detections[plate] >= 2:
                    print(f"TABLICA POTWIERDZONA: {plate} (wykryto {consecutive_detections[plate]} razy)")
                    last_detected_plate = plate
                    last_detection_time = current_time
                    consecutive_detections = {}  # Resetuj liczniki
                elif DEBUG_MODE:
                    print(f"TABLICA KANDYDAT: {plate} (wymaga potwierdzenia)")

            time.sleep(0.1)  # Krótsze opóźnienie dla lepszej responsywności
    except KeyboardInterrupt:
        print("Zatrzymano przez użytkownika")
    except Exception as e:
        print(f"Błąd podczas przetwarzania: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        print("Kamera zwolniona")

if __name__ == "__main__":
    main()