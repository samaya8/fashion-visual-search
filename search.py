"""Model loading, image embedding, cosine similarity, and Top-K retrieval."""

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def load_embedding_model():
    """Load pretrained MobileNetV3-Small and remove its classifier."""
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # MobileNet normally turns image features into 1,000 ImageNet class scores.
    # Identity makes it return the 576-number feature vector instead.
    model.classifier = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def create_embedding(image, model, preprocessing, device="cpu"):
    """Convert one PIL image into one normalized NumPy embedding."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    image = image.convert("RGB")
    image_tensor = preprocessing(image).unsqueeze(0).to(device)

    # no_grad saves memory because we are using the model, not training it.
    with torch.no_grad():
        embedding = model(image_tensor)

    embedding = embedding.cpu().numpy()[0].astype("float32")

    # Unit length makes a dot product equal cosine similarity.
    length = np.linalg.norm(embedding)
    if length > 0:
        embedding = embedding / length
    return embedding


def find_similar_images(query_embedding, dataset_embeddings, top_k=5, candidate_indices=None):
    """Return indices and cosine scores for the Top-K closest embeddings.

    Parameters:
    - query_embedding: 1D normalized query vector (D,)
    - dataset_embeddings: 2D normalized catalog matrix (N, D)
    - top_k: number of nearest results to retrieve
    - candidate_indices: optional 1D array of row indices to restrict search to
    """
    if dataset_embeddings.ndim != 2:
        raise ValueError("dataset_embeddings must be a 2D array")
    if len(dataset_embeddings) == 0:
        raise ValueError("dataset_embeddings is empty")
    if dataset_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("Query and dataset embedding sizes do not match")

    if candidate_indices is not None:
        if len(candidate_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        candidate_indices = np.asarray(candidate_indices)
        candidate_embeddings = dataset_embeddings[candidate_indices]
        similarity_scores = candidate_embeddings @ query_embedding
        k = min(top_k, len(similarity_scores))
        local_top = np.argsort(similarity_scores)[::-1][:k]
        return candidate_indices[local_top], similarity_scores[local_top]

    top_k = min(top_k, len(dataset_embeddings))

    # Embeddings are already normalized, so matrix multiplication calculates
    # cosine similarity between the query and every dataset image.
    similarity_scores = dataset_embeddings @ query_embedding
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    return top_indices, similarity_scores[top_indices]

