import os
import json
import hashlib
from PIL import Image

# 📁 Change this path to your dataset root
DATASET_PATH = "/home/mohit/doctr/references/SROIE2019"

# Output file
OUTPUT_JSON = os.path.join(DATASET_PATH, "labels.json")

labels = {}

for file in os.listdir(DATASET_PATH):
    if file.endswith(".jpg"):
        img_path = os.path.join(DATASET_PATH, file)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        if not os.path.exists(txt_path):
            print(f"⚠️ Missing annotation for {file}")
            continue

        # 1️⃣ Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # 2️⃣ Compute MD5 hash of image for traceability
        with open(img_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()

        # 3️⃣ Parse OCR txt file (box coordinates + text)
        polygons = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue  # Skip malformed lines
                try:
                    coords = list(map(int, parts[:8]))
                    polygon = [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]],
                    ]
                    polygons.append(polygon)
                except ValueError:
                    continue  # Skip lines with invalid numbers

        # 4️⃣ Add to dictionary
        labels[file] = {
            "img_dimensions": [width, height],
            "img_hash": img_hash,
            "polygons": polygons
        }

# 5️⃣ Save as JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=4)

print(f"✅ labels.json created with {len(labels)} entries at {OUTPUT_JSON}")
