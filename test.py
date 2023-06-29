import clip
import torch
import pickle
import sklearn
import time
from PIL import Image
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device)

repo_id = "adhamelarabawy/fashion_human_classifier"
model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl")

with open(model_path, 'rb') as file:
    human_classifier = pickle.load(file)

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

url = "<img url here>"
img = download_image(url).convert("RGB")

# time the prediction
start = time.time()
features = clip_model.encode_image(clip_preprocess(img).unsqueeze(0).to(device)).detach().cpu().numpy()
encode_time = time.time() - start
pred = human_classifier.predict(features) # True = has human, False = no human
pred_time = time.time() - encode_time - start

print(f"Encode time: {encode_time*1000:.3f} milliseconds")
print(f"Prediction time: {pred_time*1000:.3f} milliseconds")
print(f"Prediction (has_human): {pred}")