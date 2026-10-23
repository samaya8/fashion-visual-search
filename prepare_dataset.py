"""Create a clean teaching subset from Fashion Product Images (Small)."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import DATA_DIR, IMAGES_DIR, METADATA_PATH


DEFAULT_SOURCE_DIR = DATA_DIR / "source" / "extracted"
RANDOM_SEED = 42

# These shares keep the catalog varied without letting Apparel dominate it.
CATEGORY_SHARES = {
    "Apparel": 0.30,
    "Accessories": 0.30,
    "Footwear": 0.30,
    "Personal Care": 0.10,
}


def prepare_dataset(max_images=15000, source_dir=DEFAULT_SOURCE_DIR):
    """Select product rows, copy their images, and create metadata.csv."""
    source_dir = Path(source_dir)
    source_metadata_path = source_dir / "styles.csv"
    source_images_dir = source_dir / "images"

    if not source_metadata_path.exists() or not source_images_dir.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Read the Dataset setup section in README.md. "
            f"Expected files under: {source_dir}"
        )

    # A small number of source rows are malformed, so we safely skip only those.
    products = pd.read_csv(source_metadata_path, on_bad_lines="skip")
    products = products.dropna(
        subset=["id", "masterCategory", "subCategory", "articleType"]
    ).copy()
    products["id"] = products["id"].astype(int)

    # Keep rows that belong to the four useful catalog groups and have a JPEG.
    products = products[products["masterCategory"].isin(CATEGORY_SHARES)].copy()
    products["source_path"] = products["id"].apply(
        lambda product_id: source_images_dir / f"{product_id}.jpg"
    )
    products = products[products["source_path"].apply(Path.exists)].copy()

    max_images = min(max_images, len(products))
    selected_groups = []

    for master_category, share in CATEGORY_SHARES.items():
        group = products[products["masterCategory"] == master_category]
        requested_count = round(max_images * share)
        selected_groups.append(
            group.sample(
                n=min(requested_count, len(group)),
                random_state=RANDOM_SEED,
            )
        )

    selected = pd.concat(selected_groups).drop_duplicates(subset="id")

    # If rounding or a small group left empty places, fill them from valid rows.
    remaining_count = max_images - len(selected)
    if remaining_count > 0:
        remaining = products[~products["id"].isin(selected["id"])]
        selected = pd.concat(
            [
                selected,
                remaining.sample(
                    n=min(remaining_count, len(remaining)),
                    random_state=RANDOM_SEED,
                ),
            ]
        )

    selected = selected.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(selected):,} selected product images...")
    image_paths = []
    for index, row in selected.iterrows():
        filename = f"{row['id']}.jpg"
        output_path = IMAGES_DIR / filename
        shutil.copy2(row["source_path"], output_path)
        image_paths.append(f"data/images/{filename}")

        if (index + 1) % 1000 == 0:
            print(f"  Copied {index + 1:,} images")

    category_names = sorted(selected["subCategory"].unique())
    category_to_id = {name: index for index, name in enumerate(category_names)}

    metadata = pd.DataFrame(
        {
            "image_id": selected["id"].astype(str),
            "image_path": image_paths,
            # The original app and evaluation continue to use this simple field.
            "category": selected["subCategory"],
            "category_id": selected["subCategory"].map(category_to_id),
            # Rich fields are available for exploration and result cards.
            "master_category": selected["masterCategory"],
            "subcategory": selected["subCategory"],
            "article_type": selected["articleType"],
            "color": selected["baseColour"].fillna("Unknown"),
            "gender": selected["gender"].fillna("Unknown"),
            "season": selected["season"].fillna("Unknown"),
            "usage": selected["usage"].fillna("Unknown"),
            "product_name": selected["productDisplayName"].fillna("Unnamed product"),
            "split": "catalog",
        }
    )

    metadata.to_csv(METADATA_PATH, index=False)
    print(f"Saved metadata to: {METADATA_PATH}")
    print("\nMaster-category distribution:")
    print(metadata["master_category"].value_counts())
    print("\nSearch/evaluation categories:")
    print(metadata["category"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-images",
        type=int,
        default=15000,
        help="Number of products to include (default: 15,000)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Folder containing styles.csv and images/",
    )
    arguments = parser.parse_args()
    prepare_dataset(arguments.max_images, arguments.source_dir)

