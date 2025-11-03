import os
import json
import hashlib
from PIL import Image
from tqdm import tqdm

def create_labels_json(images_dir, annotations_dir, output_file):
    """
    Create labels.json for Doctr OCR from separate images and annotations folders.
    """
    labels = {}

    # Loop through all images
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith((".jpg", ".png")):
            # Corresponding txt file
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(annotations_dir, txt_file)

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    labels[img_file] = text
            else:
                print(f"Warning: Annotation file not found for {img_file}")

    # Save labels.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"Created labels.json with {len(labels)} entries at {output_file}")


# ======== Training data ========
train_images_dir = "/home/mohit/MS11011/doctr/references/recognition/train/images"
train_annotations_dir = "/home/mohit/MS11011/doctr/references/recognition/train/annotations"
train_output_file = "/home/mohit/MS11011/doctr/references/recognition/train/labels.json"

create_labels_json(train_images_dir, train_annotations_dir, train_output_file)

# ======== Testing data ========
test_images_dir = "/home/mohit/MS11011/doctr/references/recognition/test/images"
test_annotations_dir = "/home/mohit/MS11011/doctr/references/recognition/test/annotations"
test_output_file = "/home/mohit/MS11011/doctr/references/recognition/test/labels.json"

create_labels_json(test_images_dir, test_annotations_dir, test_output_file)

