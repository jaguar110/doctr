import os
import json
import hashlib
from PIL import Image

# 📁 Root dataset path
DATASET_PATH = "/home/mohit/MS11011/doctr/references/SROIE2019"

# 📁 Define subfolders for train and test
DATA_SPLITS = {
    "train": os.path.join(DATASET_PATH, "train"),
    "test": os.path.join(DATASET_PATH, "test"),
}

for split_name, split_path in DATA_SPLITS.items():
    img_dir = os.path.join(split_path, "img")  # or "images" if that's your folder name
    box_dir = os.path.join(split_path, "box")
    output_json = os.path.join(split_path, "labels.json")  # ✅ save inside split folder

    labels = {}

    if not os.path.exists(img_dir):
        print(f"⚠️ Skipping missing directory: {img_dir}")
        continue

    for file in os.listdir(img_dir):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(img_dir, file)
            txt_path = os.path.join(box_dir, os.path.splitext(file)[0] + ".txt")

            if not os.path.exists(txt_path):
                print(f"⚠️ Missing annotation for {file}")
                continue

            # 1️⃣ Get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size

            # 2️⃣ Compute MD5 hash
            with open(img_path, "rb") as f:
                img_hash = hashlib.md5(f.read()).hexdigest()

            # 3️⃣ Parse annotation file (handle encoding errors)
            polygons = []
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                print(f"⚠️ Encoding issue in {txt_path} — using latin-1 instead.")
                with open(txt_path, "r", encoding="latin-1") as f:
                    lines = f.readlines()

            for line in lines:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
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
                    continue

            # 4️⃣ Add entry
            labels[file] = {
                "img_dimensions": [width, height],
                "img_hash": img_hash,
                "polygons": polygons
            }

    # 5️⃣ Save labels.json in the split folder
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)

    print(f"✅ {split_name.upper()} labels.json created with {len(labels)} entries at {output_json}")
