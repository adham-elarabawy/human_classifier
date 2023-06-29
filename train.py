import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm.auto import tqdm
from datasets import load_dataset
import pickle
from huggingface_hub import HfApi

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, clip_preprocess = clip.load("ViT-B/32", device)


# Load the dataset
def preprocess(examples):
    """Preprocessing images for batches on the fly."""
    examples["image"] = [clip_preprocess(img) for img in examples["image"]]
    return examples

ds = load_dataset("adhamelarabawy/fashion_human_classification")
proc_ds = ds.with_transform(
    preprocess
)

def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=100), desc="Features encoded"):
            images = batch["image"]
            labels = batch["has_human"]
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(proc_ds["train"])
test_features, test_labels = get_features(proc_ds["test"])

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

# save scikit learn model to disk
filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

api = HfApi()
repo_name="adhamelarabawy/fashion_human_classifier"
api.create_repo(repo_name, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=repo_name,
    repo_type="model",
)