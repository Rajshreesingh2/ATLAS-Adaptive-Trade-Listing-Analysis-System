"""
ATLAS — Phase 3a: Image Downloader
Downloads 10,000 product images from Amazon URLs
Saves to data/images/ with category subfolders
"""

import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(BASE)
PROCESSED = os.path.join(ROOT, "data", "processed")
IMAGES    = os.path.join(ROOT, "data", "images")
os.makedirs(IMAGES, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
DOWNLOAD_COUNT = 10000
IMAGE_SIZE     = (224, 224)  # Standard CNN input size
WORKERS        = 8           # Parallel downloads

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def download_image(row):
    """Download a single image and save it."""
    try:
        url        = row["image_url"]
        product_id = str(row["product_id"])
        category   = str(row.get("predicted_category", "Other")).replace(" ", "_")
        
        # Create category folder
        cat_folder = os.path.join(IMAGES, category)
        os.makedirs(cat_folder, exist_ok=True)
        
        # File path
        filename = product_id.replace("/", "_") + ".jpg"
        filepath = os.path.join(cat_folder, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            return {"status": "skipped", "product_id": product_id, "path": filepath}
        
        # Download
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return {"status": "failed", "product_id": product_id, "error": f"HTTP {response.status_code}"}
        
        # Open and resize
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(filepath, "JPEG", quality=85)
        
        return {
            "status":     "success",
            "product_id": product_id,
            "category":   category,
            "path":       filepath,
            "size":       os.path.getsize(filepath)
        }
    
    except Exception as e:
        return {"status": "failed", "product_id": str(row.get("product_id", "")), "error": str(e)}


def main():
    print("=" * 60)
    print("ATLAS — Phase 3a: Image Downloader")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load NLP dataset (has predicted_category)
    nlp_path = os.path.join(PROCESSED, "atlas_nlp_dataset.csv")
    if os.path.exists(nlp_path):
        df = pd.read_csv(nlp_path, low_memory=False)
        print(f"Loaded NLP dataset: {len(df):,} records")
    else:
        df = pd.read_csv(os.path.join(PROCESSED, "atlas_master_dataset.csv"), low_memory=False)
        df["predicted_category"] = "Other"
        print(f"Loaded master dataset: {len(df):,} records")
    
    # Filter to records with valid image URLs
    df = df[df["image_url"].notna()].copy()
    df = df[df["image_url"].str.startswith("http")].copy()
    print(f"Records with valid image URLs: {len(df):,}")
    
    # Sample evenly across categories for balanced training
    categories = df["predicted_category"].value_counts()
    print(f"\nCategory distribution (top 10):")
    for cat, count in categories.head(10).items():
        print(f"  {cat:<20} {count:>8,}")
    
    # Sample up to DOWNLOAD_COUNT images balanced across categories
    samples = []
    per_category = DOWNLOAD_COUNT // len(categories)
    per_category = max(per_category, 50)
    
    for category in categories.index:
        cat_df = df[df["predicted_category"] == category]
        n      = min(len(cat_df), per_category)
        samples.append(cat_df.sample(n, random_state=42))
    
    sample_df = pd.concat(samples, ignore_index=True).head(DOWNLOAD_COUNT)
    print(f"\nDownloading {len(sample_df):,} images across {len(categories)} categories...")
    print(f"Using {WORKERS} parallel workers")
    
    # Download in parallel
    results   = []
    success   = 0
    failed    = 0
    skipped   = 0
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(download_image, row): i
            for i, row in sample_df.iterrows()
        }
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            if result["status"] == "success":   success += 1
            elif result["status"] == "skipped": skipped += 1
            else:                               failed  += 1
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1:,}/{len(sample_df):,} | "
                      f"Success: {success:,} | Failed: {failed:,}")
    
    # Save download log
    results_df  = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PROCESSED, "image_download_log.csv"), index=False)
    
    # Count downloaded images per category
    print(f"\nDownload Summary:")
    print(f"  Success:  {success:,}")
    print(f"  Skipped:  {skipped:,}")
    print(f"  Failed:   {failed:,}")
    print(f"  Total:    {len(results):,}")
    
    print(f"\nImages per category:")
    for category in os.listdir(IMAGES):
        cat_path = os.path.join(IMAGES, category)
        if os.path.isdir(cat_path):
            count = len(os.listdir(cat_path))
            print(f"  {category:<25} {count:>6,} images")
    
    # Save metadata for Colab training
    metadata = {
        "total_images":    success,
        "image_size":      IMAGE_SIZE,
        "categories":      os.listdir(IMAGES),
        "images_dir":      IMAGES,
        "downloaded_at":   datetime.now().isoformat(),
    }
    with open(os.path.join(PROCESSED, "image_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[DONE] Images saved to: data/images/")
    print(f"Next step: Upload data/images/ to Google Drive")
    print(f"Then open the Colab notebook to train CNN")
    print("=" * 60)


if __name__ == "__main__":
    main()
