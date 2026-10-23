from pathlib import Path


# Centralized project configuration for data directories and model parameters.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

# The same image size and model must be used when building the dataset embeddings
# and when processing a new query image in Streamlit.
IMAGE_SIZE = 224
DEFAULT_TOP_K = 5
MODEL_NAME = "MobileNetV3-Small (ImageNet pretrained)"

