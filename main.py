import os
import time
import cv2
from paddleocr import PaddleOCR
import numpy as np
import sys
import threading
import platform
import pathlib
import requests
from dotenv import load_dotenv

load_dotenv()

print(f"AAA {os.environ.get('DOCKER_CONTAINER')}")

# Wykrycie systemu operacyjnego
SYSTEM_OS = platform.system().lower()
IS_WINDOWS = SYSTEM_OS == 'windows'
IS_LINUX = SYSTEM_OS == 'linux'
IS_MACOS = SYSTEM_OS == 'darwin'

print(f"🖥️  Wykryto system: {platform.system()} {platform.release()}")

# Pobierz konfigurację z ENV - uniwersalną dla wszystkich systemów
CAMERA_URL = os.getenv('CAMERA_URL', '')
CAMERA_TYPE = 'usb' if CAMERA_URL.isnumeric() else 'ip'
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')

# Inicjalizacja OCR z mniejszym modelem i timeout
ocr = None

def get_camera():
    """Inicjalizuj kamerę z optymalizacjami dla różnych systemów"""

    if CAMERA_TYPE == 'usb':
        try:
            cam_index = int(CAMERA_URL)
        except ValueError:
            cam_index = 0

        # Różne backendy dla różnych systemów
        if IS_WINDOWS:
            # Windows: DirectShow jest często najlepszy
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback na domyślny backend
                cap = cv2.VideoCapture(cam_index)
        elif IS_LINUX:
            # Linux: V4L2 jest native
            cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                # Fallback na domyślny backend
                cap = cv2.VideoCapture(cam_index)
        elif IS_MACOS:
            # macOS: AVFoundation
            cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                # Fallback na domyślny backend
                cap = cv2.VideoCapture(cam_index)
        else:
            # Nieznany system - użyj domyślnego
            cap = cv2.VideoCapture(cam_index)

    elif CAMERA_TYPE == 'ip':
        cap = cv2.VideoCapture(CAMERA_URL)
    else:
        raise ValueError('Nieznany typ kamery: {}'.format(CAMERA_TYPE))

    if not cap.isOpened():
        error_msg = f'Nie można otworzyć kamery!'
        if CAMERA_TYPE == 'usb':
            error_msg += f'\n💡 Sprawdź:'
            if IS_WINDOWS:
                error_msg += f'\n  - Czy kamera jest podłączona i rozpoznana w Device Manager'
                error_msg += f'\n  - Czy żadna inna aplikacja nie używa kamery'
            elif IS_LINUX:
                error_msg += f'\n  - ls /dev/video* (sprawdź dostępne urządzenia)'
                error_msg += f'\n  - Uprawnienia użytkownika do grupy video'
            elif IS_MACOS:
                error_msg += f'\n  - Uprawnienia do kamery w System Preferences > Privacy'
        raise RuntimeError(error_msg)

    # OPTYMALIZACJE DLA AKTUALNOŚCI KLATEK
    # Ustaw mały bufor aby zawsze pobierać najnowsze klatki
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"📹 Kamera otwarta ({CAMERA_TYPE}): {CAMERA_URL}")
    return cap

def optimize_camera_for_system(cap):
    """Zastosuj optymalizacje kamery specyficzne dla systemu operacyjnego"""
    optimizations_applied = 0

    try:
        # Optymalizacje wspólne dla wszystkich systemów
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # OPTYMALIZACJA 1: Ustaw rozdzielczość dla wydajności
        if cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720):
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  ✓ Rozdzielczość: {width:.0f}x{height:.0f}")
            optimizations_applied += 1

        # OPTYMALIZACJA 2: FPS
        if cap.set(cv2.CAP_PROP_FPS, 30):
            new_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  ✓ FPS: {original_fps:.1f} -> {new_fps:.1f}")
            optimizations_applied += 1

        # Optymalizacje specyficzne dla systemu
        if IS_WINDOWS:
            # Windows-specific optimizations
            if hasattr(cv2, 'CAP_PROP_BUFFER_SIZE'):
                if cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1):
                    print("  ✓ Mały bufor kamery (Windows)")
                    optimizations_applied += 1

        elif IS_LINUX:
            # Linux-specific optimizations
            if hasattr(cv2, 'CAP_PROP_BUFFER_SIZE'):
                if cap.set(cv2.CAP_PROP_BUFFER_SIZE, 2):  # Linux może potrzebować większego bufora
                    print("  ✓ Zoptymalizowany bufor kamery (Linux)")
                    optimizations_applied += 1

            # Na Linux często można ustawić fourcc dla lepszej wydajności
            if hasattr(cap, 'set') and hasattr(cv2, 'VideoWriter_fourcc'):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                print("  ✓ Ustawiono MJPEG codec (Linux)")
                optimizations_applied += 1

        elif IS_MACOS:
            # macOS-specific optimizations
            # Na macOS często nie można ustawić BUFFER_SIZE, więc pomijamy
            print("  ✓ Konfiguracja dla macOS")
            optimizations_applied += 1

        # OPTYMALIZACJA 3: Wyłącz auto-exposure jeśli dostępne
        if hasattr(cv2, 'CAP_PROP_AUTO_EXPOSURE'):
            # Różne systemy mogą mieć różne wartości dla wyłączenia auto-exposure
            if IS_LINUX:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Linux
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Windows/macOS
            print("  ✓ Wyłączono auto-exposure")
            optimizations_applied += 1

        print(f"🚀 Zastosowano {optimizations_applied} optymalizacji dla {platform.system()}")

    except Exception as e:
        print(f"⚠️ Niektóre optymalizacje kamery niedostępne: {e}")
        print("  📝 System będzie działał z domyślnymi ustawieniami")

def init_ocr():
    global ocr
    if ocr is None:
        try:
            # Sprawdź instalację przed inicjalizacją
            print("🔍 Sprawdzanie instalacji PaddlePaddle...")
            try:
                import paddle
                import paddleocr
                print(f"✅ PaddlePaddle: {paddle.__version__}")
                print(f"✅ PaddleOCR: {paddleocr.__version__}")

                # Test podstawowej funkcjonalności
                print("🔍 Test GPU/CUDA...")
                print(f"   paddle.device.get_device(): {paddle.device.get_device()}")
                try:
                    cuda_count = paddle.device.cuda.device_count()
                    print(f"   paddle.device.cuda.device_count(): {cuda_count}")
                except Exception as e:
                    print(f"   ⚠️  paddle.device.cuda.device_count() error: {e}")

            except ImportError as e:
                print(f"❌ Błąd importu: {e}")
                print("💡 Spróbuj: pip install paddlepaddle-gpu paddleocr")
                return
            except Exception as e:
                print(f"❌ Błąd testowania: {e}")

            print("🔧 Inicjalizacja PaddleOCR...")
            print("UWAGA: Pierwsze uruchomienie może trwać kilka minut - pobieranie modeli...")

            # Sprawdź dostępność GPU na różnych systemach
            use_gpu = False
            gpu_info = "CPU"

            try:
                import paddle
                print(f"✅ PaddlePaddle zaimportowany: {paddle.__version__}")

                # Nowy sposób detekcji GPU
                device_info = paddle.device.get_device()
                cuda_count = 0
                try:
                    cuda_count = paddle.device.cuda.device_count()
                except Exception as e:
                    print(f"⚠️  paddle.device.cuda.device_count() error: {e}")

                gpu_available = device_info.startswith('gpu') or cuda_count > 0

                if gpu_available:
                    use_gpu = True
                    if IS_LINUX:
                        gpu_info = "GPU/CUDA (Linux)"
                    elif IS_WINDOWS:
                        gpu_info = "GPU/CUDA (Windows)"
                    elif IS_MACOS:
                        gpu_info = "GPU/Metal (macOS)"
                    else:
                        gpu_info = "GPU/CUDA"
                    print(f"🚀 Wykryto GPU! {gpu_info}")
                else:
                    use_gpu = False
                    if IS_WINDOWS:
                        gpu_info = "CPU (Windows - brak GPU/CUDA)"
                    elif IS_LINUX:
                        gpu_info = "CPU (Linux - brak GPU/CUDA)"
                    elif IS_MACOS:
                        gpu_info = "CPU (macOS - brak GPU/Metal)"
                    else:
                        gpu_info = "CPU (brak GPU)"
                    print(f"🔧 Używam: {gpu_info}")
            except ImportError as import_e:
                print(f"❌ Błąd importu PaddlePaddle: {import_e}")
                print("🔧 Używam: CPU (błąd importu Paddle)")
            except Exception as e:
                print(f"❌ Nieoczekiwany błąd PaddlePaddle: {e}")
                print("🔧 Używam: CPU (nie można sprawdzić Paddle)")

            # PaddleOCR inicjalizacja uniwersalna dla wszystkich systemów
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False,  # wyłącz verbose aby zmniejszyć warnings
                drop_score=0.3,  # obniż próg pewności dla lepszej detekcji
                # Użyj systemowego katalogu cache
                use_pdserving=False,  # Wyłącz serwisy dla lepszej kompatybilności
                enable_mkldnn=not IS_MACOS  # MKLDNN może nie działać na macOS
            )

            print(f"✅ PaddleOCR gotowy na {platform.system()}!")

        except ImportError as e:
            print(f"❌ Błąd importu PaddleOCR: {e}")
            print("📦 Zainstaluj wymagane pakiety:")
            if IS_WINDOWS:
                print("   pip install paddleocr paddlepaddle")
                print("   # Dla GPU na Windows: pip install paddlepaddle-gpu")
                print("   # CUDA 11.8: pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
                print("   # CUDA 12.6: pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
            elif IS_LINUX:
                print("   pip install paddleocr paddlepaddle")
                print("   # Dla GPU na Windows/Linux: pip install paddlepaddle-gpu")
                print("   # CUDA 11.8: pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
                print("   # CUDA 12.6: pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
            elif IS_MACOS:
                print("   pip install paddleocr paddlepaddle")
            ocr = None
        except Exception as e:
            print(f"❌ Błąd inicjalizacji OCR: {e}")
            print("🌐 Sprawdź połączenie internetowe - potrzebne do pobrania modeli")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            ocr = None

def preprocess_image_for_ocr(frame):
    """Ulepsz obraz przed OCR - specjalnie dla tablic rejestracyjnych"""
    try:
        # Konwersja do skali szarości jeśli kolorowy
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Zwiększ kontrast dla lepszej czytelności
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Filtruj szum
        denoised = cv2.medianBlur(enhanced, 3)

        # Konwertuj z powrotem do BGR dla PaddleOCR
        bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return bgr
    except Exception as e:
        if DEBUG_MODE:
            print(f"Błąd przetwarzania obrazu: {e}")
        return frame

def detect_and_read_plate(frame):
    global ocr
    if ocr is None:
        return None

    try:
        # Ulepsz obraz przed OCR
        processed_frame = preprocess_image_for_ocr(frame)

        # Uruchom OCR z klasyfikacją kąta dla lepszej detekcji
        result = ocr.ocr(processed_frame, cls=True)

        if result and len(result) > 0 and result[0] is not None:
            if DEBUG_MODE:
                print(f"OCR wykrył {len(result[0])} tekstów:")

            best_candidates = []

            # Przeszukaj wszystkie wykryte teksty
            for detection in result[0]:
                if len(detection) >= 2:
                    text = detection[1][0].strip()  # detection[1][0] to tekst w PaddleOCR
                    confidence = detection[1][1]     # detection[1][1] to pewność

                    if DEBUG_MODE:
                        print(f"  '{text}' (pewność: {confidence:.2f})")

                    # Wyczyść tekst - usuń spacje i znaki specjalne
                    import re
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                    # Filtruj potencjalne tablice rejestracyjne z niższym progiem
                    if clean_text and len(clean_text) >= 4 and confidence > 0.3:
                        if is_license_plate(clean_text):
                            best_candidates.append((clean_text, confidence))
                            if DEBUG_MODE:
                                print(f"  -> Kandydat: '{clean_text}' (pewność: {confidence:.2f})")

            # Zwróć najlepszy kandydat
            if best_candidates:
                best_candidates.sort(key=lambda x: x[1], reverse=True)
                return best_candidates[0][0]

        elif DEBUG_MODE:
            print("OCR nie wykrył żadnego tekstu")

    except Exception as e:
        print(f"Błąd OCR: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return None
    return None

def is_license_plate(text):
    """Sprawdź czy tekst może być tablicą rejestracyjną"""
    import re

    # Usuń spacje i znaki specjalne
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Sprawdź długość (tablice zwykle 4-8 znaków)
    if len(clean_text) < 4 or len(clean_text) > 8:
        return False

    # Sprawdź czy zawiera cyfry i litery (typowe dla tablic)
    has_letters = bool(re.search(r'[A-Z]', clean_text))
    has_numbers = bool(re.search(r'[0-9]', clean_text))

    return has_letters and has_numbers

def test_system():
    """Test systemu - uniwersalny dla Windows/Linux/macOS"""
    print(f"=== TEST SYSTEMU {platform.system().upper()} ===")

    # Test OpenCV
    try:
        import cv2
        opencv_version = cv2.__version__
        print(f"✓ OpenCV {opencv_version} - OK")

        # Sprawdź dostępne właściwości kamery (różne na różnych systemach)
        available_props = []
        test_props = [
            ('BUFFER_SIZE', 'CAP_PROP_BUFFER_SIZE'),
            ('FPS', 'CAP_PROP_FPS'),
            ('FRAME_WIDTH', 'CAP_PROP_FRAME_WIDTH'),
            ('FRAME_HEIGHT', 'CAP_PROP_FRAME_HEIGHT'),
            ('AUTO_EXPOSURE', 'CAP_PROP_AUTO_EXPOSURE')
        ]

        for prop_name, prop_attr in test_props:
            if hasattr(cv2, prop_attr):
                available_props.append(prop_name)

        if available_props:
            print(f"  ✓ Dostępne właściwości kamery: {', '.join(available_props)}")
        else:
            print(f"  ⚠️ Ograniczone właściwości kamery (OpenCV {opencv_version})")

    except ImportError:
        print("✗ OpenCV - BRAK")
        print("📦 Instalacja:")
        if IS_WINDOWS:
            print("   pip install opencv-python")
        elif IS_LINUX:
            print("   pip install opencv-python")
            print("   # Lub: sudo apt install python3-opencv")
        elif IS_MACOS:
            print("   pip install opencv-python")
            print("   # Lub: brew install opencv")
        return False

    # Test PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("✓ PaddleOCR import - OK")
    except ImportError:
        print("✗ PaddleOCR - BRAK")
        print("Zainstaluj: pip install paddleocr paddlepaddle")
        return False

    # Test numpy
    try:
        import numpy as np
        print("✓ NumPy - OK")
    except ImportError:
        print("✗ NumPy - BRAK")
        return False

    print("✓ Wszystkie wymagane biblioteki dostępne")
    return True

def send_to_webhook(plate):
    """Wysyła dane do webhooku"""

    if WEBHOOK_URL == "":
        print(f"Pomijam wysyłanie do webhooku - WEBHOOK_URL jest pusty")
        return

    try:
        response = requests.post(WEBHOOK_URL, json={"plate": plate})
        if response.status_code == 200:
            print(f"✅ Wysłano do webhooku: {plate}")
        else:
            print(f"❌ Błąd przy wysyłaniu do webhooku: {response.status_code}")
    except Exception as e:
        print(f"❌ Błąd przy wysyłaniu do webhooku: {e}")

def main():
    print("=== SYSTEM ROZPOZNAWANIA TABLIC REJESTRACYJNYCH ===")
    print(f"🌍 Uniwersalny system dla {platform.system()}")
    print()

    # Test systemu - sprawdź kompatybilność
    if not test_system():
        print(f"\n❌ Błąd: Nie wszystkie wymagane komponenty są dostępne na {platform.system()}!")
        print("📦 Zainstaluj brakujące pakiety i spróbuj ponownie.")
        return

    print()

    # Inicjalizacja OCR
    print("🔧 Inicjalizacja PaddleOCR...")
    init_ocr()

    if ocr is None:
        print("❌ Błąd: Nie można zainicjalizować PaddleOCR!")
        print("🌐 Sprawdź połączenie internetowe - potrzebne do pobrania modeli")
        return

    print("✅ PaddleOCR zainicjalizowany")
    print("📹 Łączę z kamerą...")

    # Połączenie z kamerą
    try:
        cap = get_camera()

        # Optymalizacje kamery specyficzne dla systemu
        print("🔧 Konfiguruję optymalizacje kamery...")
        optimize_camera_for_system(cap)

    except Exception as e:
        print(f"❌ Błąd połączenia z kamerą: {e}")
        return

    print()
    print(f"🚗 SYSTEM AKTYWNY - {platform.system()} - rozpoznawanie tablic...")
    print("⌨️  Naciśnij Ctrl+C aby zatrzymać")
    print("-" * 50)

    try:
        analysis_running = False

        while True:
            if (not analysis_running):
                # OPRÓŻNIJ BUFOR - pobieraj klatki przez 0.5 sekundy
                start_flush = time.time()
                flushed = 0
                while time.time() - start_flush < 0.5:
                    cap.grab()
                    flushed += 1
                if DEBUG_MODE:
                    print(f"[DEBUG] Opróżniono bufor, odrzucono {flushed} klatek")

                # POCZEKAJ NA NOWĄ KLATKĘ
                time.sleep(0.05)  # 50 ms
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Brak obrazu z kamery! Próbuję ponownie za 2 sekundy...")
                    time.sleep(2)
                    continue

                if DEBUG_MODE:
                    print(f"[{time.strftime('%H:%M:%S')}] Rozpoczynam analizę klatki...")

                analysis_running = True

                analysis_started_at = time.time()
                # Rozpoznawanie tablicy
                plate = detect_and_read_plate(frame)

                analysis_ended_at = time.time()
                analysis_duration = analysis_ended_at - analysis_started_at

                analysis_running = False  # Zakończ flagę analizy

                if plate:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"🎯 [{timestamp}] WYKRYTO TABLICĘ: {plate} w ciągu {analysis_duration:.2f} sekund")
                    send_to_webhook(plate)

    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("🛑 Zatrzymano przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd podczas przetwarzania: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
    finally:
        try:
            cap.release()
            print("📷 Kamera zwolniona")
        except:
            pass
        print(f"👋 Do widzenia z {platform.system()}!")

if __name__ == "__main__":
    main()