import os
import json
import hashlib
from PIL import Image

# === Paths ===
# fro training data
annotations_dir = "/home/mohit/13 October/OCR_Detection_Training/training_data/annotations"
images_dir = "/home/mohit/13 October/OCR_Detection_Training/training_data/images"
output_file = "/home/mohit/13 October/OCR_Detection_Training/training_data/labels.json"

# for testing data
annotations_dir = "/home/mohit/13 October/OCR_Detection_Training/testing_data/annotations"
images_dir = "/home/mohit/13 October/OCR_Detection_Training/testing_data/images"
output_file = "/home/mohit/13 October/OCR_Detection_Training/testing_data/labels.json"

merged = {}

def compute_sha256(img_path):
    """Compute SHA256 hash of an image file"""
    with open(img_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# === Loop through each image annotation file ===
for file_name in os.listdir(annotations_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(annotations_dir, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)

        # Infer image file name (FUNSD JSONs match image base names)
        img_name = file_name.replace(".json", ".png")
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"⚠️ Image not found for {file_name}, skipping.")
            continue

        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # Compute image hash
        img_hash = compute_sha256(img_path)

        # === Extract polygons ===
        polygons = []
        if "form" in data:  # FUNSD format check
            for field in data["form"]:
                if "words" in field:
                    for word in field["words"]:
                        if "box" in word:
                            x1, y1, x2, y2 = word["box"]
                            polygons.append([
                                [x1, y1],
                                [x2, y1],
                                [x2, y2],
                                [x1, y2]
                            ])

        merged[img_name] = {
            "img_dimensions": [width, height],
            "img_hash": img_hash,
            "polygons": polygons
        }

# === Write final labels.json ===
with open(output_file, "w") as f:
    json.dump(merged, f, indent=2)

print(f"✅ Merged {len(merged)} annotation files into {output_file}")
