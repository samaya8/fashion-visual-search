"""Generate and save all dataset embeddings once, before running the app."""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import EMBEDDINGS_PATH, METADATA_PATH, PROJECT_DIR
from search import load_embedding_model


class ImageFileDataset(Dataset):
    """Small adapter that lets a DataLoader read image paths from metadata."""

    def __init__(self, metadata, preprocessing):
        self.metadata = metadata
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        relative_path = self.metadata.iloc[index]["image_path"]
        image_path = PROJECT_DIR / relative_path
        with Image.open(image_path) as image:
            return self.preprocessing(image.convert("RGB"))


def generate_embeddings(batch_size=64):
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "metadata.csv was not found. Run: python prepare_dataset.py"
        )

    metadata = pd.read_csv(METADATA_PATH)
    if metadata.empty:
        raise ValueError("metadata.csv has no rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocessing = load_embedding_model()
    model = model.to(device)

    image_dataset = ImageFileDataset(metadata, preprocessing)
    data_loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Most reliable choice for beginner Windows laptops.
    )

    embedding_batches = []
    start_time = time.time()

    for batch_number, image_batch in enumerate(data_loader, start=1):
        image_batch = image_batch.to(device)
        with torch.no_grad():
            batch_embeddings = model(image_batch)
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings, p=2, dim=1
            )

        embedding_batches.append(batch_embeddings.cpu().numpy().astype("float32"))

        if batch_number % 20 == 0 or batch_number == len(data_loader):
            processed = min(batch_number * batch_size, len(metadata))
            print(f"Processed {processed:,} / {len(metadata):,} images")

    embeddings = np.vstack(embedding_batches)
    np.save(EMBEDDINGS_PATH, embeddings)

    elapsed = time.time() - start_time
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Finished in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()
    generate_embeddings(arguments.batch_size)

