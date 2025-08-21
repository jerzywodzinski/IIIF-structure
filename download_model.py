from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
SAVE_DIRECTORY = "clip-model"

print(f"Pobieranie modelu {MODEL_ID}...")
CLIPModel.from_pretrained(MODEL_ID).save_pretrained(SAVE_DIRECTORY)
print(f"Pobieranie procesora dla {MODEL_ID}...")
CLIPProcessor.from_pretrained(MODEL_ID).save_pretrained(SAVE_DIRECTORY)

print(f"Model i procesor zostały zapisane w folderze '{SAVE_DIRECTORY}'.")