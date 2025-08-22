import io
import json
import logging
import requests
import argparse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_PATH = "clip-model"
clip_model = None
clip_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global clip_model, clip_processor
    try:
        logger.info(f"Rozpoczynam ładowanie modelu z: {MODEL_PATH}")
        clip_model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        logger.info(f"Model załadowany pomyślnie, działa na: {device.upper()}")
        return True
    except Exception as e:
        logger.critical(f"KRYTYCZNY BŁĄD: Nie udało się załadować modelu z folderu '{MODEL_PATH}'.")
        logger.exception(e)
        return False

def classify(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        texts = [
            "a photo of a newspaper cover with a title and masthead",
            "a photo of an internal page with articles and blocks of body text (not title and masthead)",
            "a photo of an internal page full of advertisements or announcements (not title and masthead)",
            "a photo of an internal page with a large illustration or photograph (not title and masthead)",
            "a photo of a table of contents or an editorial page (not title and masthead)"
        ]

        inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            is_cover = logits_per_image.argmax(-1).item() == 0
            
        return {"is_cover": bool(is_cover)}
    except Exception:
        raise

def get_full_image_url(canvas: dict, size: str = "1200,") -> str:
    try:
        service_id = canvas['images'][0]['resource']['service']['@id']
        return f"{service_id.rstrip('/')}/full/{size}/0/default.jpg"
    except (KeyError, IndexError):
        return None

def get_id_from_canvas(canvas: dict) -> str:
    try:
        service_id = canvas['images'][0]['resource']['service']['@id']
        return service_id.strip('/').split('/')[-1]
    except (KeyError, IndexError):
        return None

def analyze_manifest(manifest_url: str, is_split_scan: bool, start_page: int, end_page: int):
    logger.info(f"\nPobieranie manifestu z: {manifest_url}")
    try:
        response = requests.get(manifest_url, timeout=30)
        response.raise_for_status()
        manifest = response.json()
        canvases = manifest.get('sequences', [{}])[0].get('canvases', [])
        total_pages = len(canvases)
        logger.info(f"Znaleziono {total_pages} stron w manifeście.")
    except Exception as e:
        logger.error(f"Błąd podczas pobierania lub parsowania manifestu: {manifest_url}")
        logger.exception(e)
        return [], None

    if not canvases:
        logger.warning("W manifeście nie znaleziono żadnych stron (canvases). Zakończono analizę.")
        return [], None

    start_index = start_page - 1
    end_index = end_page if end_page is not None else total_pages
    
    if not (0 <= start_index < total_pages and start_index < end_index <= total_pages):
        logger.error(f"Nieprawidłowy zakres stron. Podano od {start_page} do {end_page}, a dostępnych jest {total_pages}.")
        return [], None
        
    canvases_to_analyze = canvases[start_index:end_index]
    logger.info(f"Rozpoczynam analizę stron od {start_index + 1} do {end_index}")
    if is_split_scan:
        logger.info("Tryb analizy: Skan dwustronicowy (obrazy będą dzielone na pół)")

    cover_pages_indices = []
    
    for i, canvas in enumerate(tqdm(canvases_to_analyze, desc="Analiza stron")):
        current_page_index = start_index + i
        image_url = get_full_image_url(canvas, size="1200,")
        if not image_url:
            logger.warning(f"Brak URL obrazu dla strony {current_page_index + 1}. Pomijam.")
            continue
        
        try:
            response = requests.get(image_url, timeout=45)
            response.raise_for_status()
            image_bytes = response.content

            is_cover_found = False
            if not is_split_scan:
                result = classify(image_bytes)
                if result.get("is_cover"):
                    is_cover_found = True
            else:
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                mid_point = width // 2
                
                left_img_pil = image.crop((0, 0, mid_point, height))
                with io.BytesIO() as left_bytes_io:
                    left_img_pil.save(left_bytes_io, format='JPEG')
                    result_left = classify(left_bytes_io.getvalue())
                
                right_img_pil = image.crop((mid_point, 0, width, height))
                with io.BytesIO() as right_bytes_io:
                    right_img_pil.save(right_bytes_io, format='JPEG')
                    result_right = classify(right_bytes_io.getvalue())
                
                if result_left.get("is_cover") or result_right.get("is_cover"):
                    is_cover_found = True
            
            if is_cover_found:
                cover_pages_indices.append(current_page_index)

        except Exception as e:
            logger.error(f"Błąd przy przetwarzaniu strony {current_page_index + 1} (URL: {image_url})")
            logger.exception(e)

    return cover_pages_indices, canvases

def setup_logging():
    logger.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler('app.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Analizuje manifest IIIF, znajduje okładki gazet i tworzy plik TXT z zakresami stron.")
    parser.add_argument("url", type=str, help="Pełny URL do pliku manifest.json")
    parser.add_argument("-o", "--output", type=str, default="output.txt", help="Nazwa pliku wyjściowego (domyślnie: output.txt)")
    parser.add_argument("--split-scan", action="store_true", help="Użyj, jeśli skany zawierają dwie strony obok siebie (będą dzielone na pół).")
    parser.add_argument("--start", type=int, default=1, help="Numer strony, od której rozpocząć analizę.")
    parser.add_argument("--end", type=int, default=None, help="Numer strony, na której zakończyć analizę (włącznie). Domyślnie do końca.")
    
    args = parser.parse_args()
    logger.info(f"Aplikacja uruchomiona z argumentami: URL={args.url}, Output={args.output}, SplitScan={args.split_scan}, Start={args.start}, End={args.end}")

    if not load_model():
        exit(1)
    
    cover_indices, all_canvases = analyze_manifest(args.url, args.split_scan, args.start, args.end)
    
    logger.info("")
    if not cover_indices:
        logger.info("Nie znaleziono żadnych okładek. Plik wyjściowy nie zostanie utworzony.")
        return

    logger.info(f"Znaleziono {len(cover_indices)} okładek:")
    for index in sorted(cover_indices):
        logger.info(f"- Strona: {index + 1}")

    logger.info(f"\nGenerowanie pliku wyjściowego: {args.output}")

    total_pages = len(all_canvases)
    
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, start_idx in enumerate(sorted(cover_indices)):
                start_id = get_id_from_canvas(all_canvases[start_idx])
                
                if i + 1 < len(cover_indices):
                    end_idx = sorted(cover_indices)[i+1] - 1
                else:
                    end_idx = total_pages - 1
                
                if end_idx < start_idx:
                    end_idx = start_idx
                    
                end_id = get_id_from_canvas(all_canvases[end_idx])
                
                if start_id and end_id:
                    f.write(f"{start_id},{end_id}\n")
        logger.info(f"Pomyślnie zapisano wyniki do pliku: {args.output}")
    except Exception as e:
        logger.error(f"Nie udało się zapisać pliku wyjściowego: {args.output}")
        logger.exception(e)

    logger.info("Zakończono. Aplikacja kończy działanie.")

if __name__ == "__main__":
    main()