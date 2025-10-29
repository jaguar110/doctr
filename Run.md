# 🧠 Document Text Detection using Doctr (PyTorch)

This project demonstrates **training and inference** for text detection models using the [Mindee Doctr](https://github.com/mindee/doctr) library in PyTorch.  
You can train on datasets like **FUNSD** and **SROIE**, and run inference on custom images to detect text regions.

---

## 🚀 Features

- Train text detection models (`db_resnet50`)
- Support for **FUNSD** and **SROIE** datasets
- Run inference on single or multiple images
- Output:
  - Bounding boxes + confidence scores (`detection_output.json`)
  - Visualization with text regions drawn (`detection_visualization.png`)

---

## 📁 Project Structure

.
├── references/
│ ├── detection/
│ │ ├── train.py # Training script
│ │ └── funsd_infer.py # Inference script
│ ├── training_data/ # FUNSD training set
│ └── testing_data/ # FUNSD validation set
├── SROIE2019/
│ ├── train/ # SROIE training set
│ └── test/ # SROIE test set
└── db_resnet50_YYYYMMDD-HHMMSS.pt # Trained model file


---

## 🧩 Environment Setup

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies
pip install python-doctr[torch] matplotlib numpy torch torchvision


🔹 Train on FUNSD Dataset

python references/detection/train.py db_resnet50 \
  --train_path references/training_data \
  --val_path references/testing_data \
  --epochs 5

🔹 Train on SROIE Dataset

python references/detection/train.py db_resnet50 \
  --train_path references/SROIE2019/train \
  --val_path references/SROIE2019/test \
  --epochs 5

⚠️ Note for SROIE:
This dataset can cause Out of Memory (OOM) errors during training.
To fix this:

@ Reduce the batch size by adding --batch_size 2 or --batch_size 4

@ Or resize images before training

Example: python references/detection/train.py db_resnet50 \
        --train_path references/SROIE2019/train \
        --val_path references/SROIE2019/test \
        --epochs 5 \
        --batch_size 2

🔍 Inference

After training, you can run inference using your trained model file.

python funsd_infer.py

🧾 Output Files
detection_output.json
detection_visualization.png