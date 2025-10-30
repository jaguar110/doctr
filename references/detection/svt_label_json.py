import os
import json
import hashlib
from xml.etree import ElementTree as ET
from PIL import Image

# 📁 Change to your SVT dataset path
DATASET_PATH = "/home/mohit/13October/doctr/references/SVT"
XML_PATH = os.path.join(DATASET_PATH, "train.xml")
IMAGES_DIR = os.path.join(DATASET_PATH, "img")
OUTPUT_JSON = os.path.join(DATASET_PATH, "labels.json")

# Parse the XML file
tree = ET.parse(XML_PATH)
root = tree.getroot()

labels = {}

for image_tag in root.findall("image"):
    image_name = image_tag.find("imageName").text.strip()
    img_path = os.path.join(DATASET_PATH, image_name)
    
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {image_name}")
        continue

    # Get image dimensions
    with Image.open(img_path) as img:
        width, height = img.size

    # Compute MD5 hash
    with open(img_path, "rb") as f:
        img_hash = hashlib.md5(f.read()).hexdigest()

    polygons = []
    tagged_rects = image_tag.find("taggedRectangles")

    if tagged_rects is not None:
        for rect in tagged_rects.findall("taggedRectangle"):
            try:
                x = int(float(rect.attrib["x"]))
                y = int(float(rect.attrib["y"]))
                w = int(float(rect.attrib["width"]))
                h = int(float(rect.attrib["height"]))

                polygon = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                polygons.append(polygon)
            except (ValueError, KeyError):
                continue

    labels[os.path.basename(image_name)] = {
        "img_dimensions": [width, height],
        "img_hash": img_hash,
        "polygons": polygons
    }

# Save JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=4)

print(f"✅ labels.json created with {len(labels)} entries at {OUTPUT_JSON}")
