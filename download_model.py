from huggingface_hub import hf_hub_download
import os
from tqdm import tqdm

MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SAVE_DIRECTORY = "clip-model"

os.makedirs(SAVE_DIRECTORY, exist_ok=True)

files_to_download = [
    "config.json",
    "pytorch_model.bin",  
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt"
]


for filename in tqdm(files_to_download, desc="Pobierane pliki"):
    try:
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            local_dir=SAVE_DIRECTORY,
            local_dir_use_symlinks=False,  
            resume_download=True         
        )
    except Exception as e:
        print(f"\nBŁĄD: Nie udało się pobrać pliku {filename}. Szczegóły: {e}")
        exit(1) 


