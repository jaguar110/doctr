import torch
import numpy as np
from doctr.models import detection
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

# -----------------------------
# 1️⃣ Input image or folder
# -----------------------------
input_path = "/home/mohit/MS11011/doctr/references/testing_data/images/82253362_3364.png"  # or folder with images

# Collect image paths
if os.path.isdir(input_path):
    image_paths = sorted([
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
else:
    image_paths = [input_path]

# -----------------------------
# 2️⃣ Load images safely
# -----------------------------
# This guarantees we always have a list of NumPy arrays
images_np = []
for img_path in image_paths:
    doc = DocumentFile.from_images(img_path)
    if isinstance(doc, list):
        # Older doctr versions may return a list
        doc = doc[0]
    if hasattr(doc, "as_images"):
        imgs = doc.as_images()
    elif isinstance(doc, np.ndarray):
        imgs = [doc]
    else:
        raise TypeError(f"Unexpected type for document: {type(doc)}")
    images_np.extend(imgs)

print(f"✅ Loaded {len(images_np)} image(s)")



# -----------------------------
# 2️⃣ Load trained detection model
# -----------------------------
model = detection.db_resnet50(pretrained=False)
model.load_state_dict(torch.load("/home/mohit/MS11011/doctr/db_resnet50_20251029-112636.pt", map_location="cpu"))
model.eval()

# -----------------------------
# 3️⃣ Convert document to tensor
# -----------------------------

def pad_to_multiple_of_32(img: np.ndarray) -> np.ndarray:
    """Pad image (H, W, C) so that both H and W are divisible by 32."""
    h, w, c = img.shape
    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)
    padded = np.zeros((new_h, new_w, c), dtype=img.dtype)
    padded[:h, :w, :] = img
    return padded

# doctr expects a tensor of shape (N, 3, H, W)

# images = images_np
# images = [torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float() / 255. for img in images]
# image_tensor = torch.cat(images, dim=0)

# Pad images before tensor conversion
padded_images = [pad_to_multiple_of_32(img) for img in images_np]
images = [torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).float() / 255. for img in padded_images]
image_tensor = torch.cat(images, dim=0)  # ✅ Combine into single batch
print(f"Image tensor shape: {image_tensor.shape}")

# -----------------------------
# 4️⃣ Run inference
# -----------------------------
with torch.no_grad():
    preds = model(image_tensor)

print("🔍 Model output type:", type(preds))
print("🔍 Keys:", preds.keys() if isinstance(preds, dict) else None)
print("🔍 Raw preds content:")
print(preds['preds'][0])
# -----------------------------
# 5️⃣ Extract boxes and scores
# -----------------------------
# Extract predictions from dict
# if isinstance(preds, dict) and 'preds' in preds:
#     preds_list = preds['preds']
#     if len(preds_list) == 0:
#         print("⚠️ No predictions found.")
#         boxes, scores = np.array([]), np.array([])
#     else:
#         preds_dict = preds_list[0]
#         boxes = preds_dict.get('boxes', torch.empty((0, 4))).detach().numpy()
#         scores = preds_dict.get('scores', torch.empty((0,))).detach().numpy()
# else:
#     raise ValueError("Unexpected model output structure.")


# -----------------------------
# 6️⃣ Save JSON results
# -----------------------------
# output_data = []
# for box, score in zip(boxes, scores):
#     x_min, y_min, x_max, y_max = box.tolist()
#     output_data.append({
#         "box": [x_min, y_min, x_max, y_max],
#         "score": float(score)
#     })

# with open("detection_output.json", "w") as f:
#     json.dump(output_data, f, indent=2)

# print("✅ Detection results saved to detection_output.json")

# # -----------------------------
# # 7️⃣ Visualize detections
# # -----------------------------
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(images_np[0]) 
# for box in boxes:
#     x_min, y_min, x_max, y_max = box
#     rect = plt.Rectangle(
#         (x_min, y_min),
#         x_max - x_min,
#         y_max - y_min,
#         fill=False,
#         color='red',
#         linewidth=2
#     )
#     ax.add_patch(rect)

# plt.axis("off")
# plt.tight_layout()
# plt.savefig("detection_visualization.png", bbox_inches="tight")
# print("✅ Visualization saved as detection_visualization.png")
# plt.show()




# image_path = "/home/mohit/MS11011/doctr/references/testing_data/images/82253362_3364.png"
# model_path = "/home/mohit/MS11011/doctr/db_resnet50_20251029-112636.pt"


# -----------------------------
# 6️⃣ Extract predictions (Doctr format)
# -----------------------------
preds_list = preds.get('preds', [])
results = []

if len(preds_list) == 0:
    print("⚠️ No predictions returned by model.")
else:
    preds_dict = preds_list[0]
    if "words" in preds_dict:
        words = preds_dict["words"]

        # Convert normalized coordinates → absolute pixels
        img_h, img_w = images_np[0].shape[:2]
        boxes = np.array([
            [
                w[0] * img_w,  # x_min
                w[1] * img_h,  # y_min
                w[2] * img_w,  # x_max
                w[3] * img_h,  # y_max
            ]
            for w in words
        ])
        scores = np.array([w[4] for w in words])

        # Filter boxes by score threshold
        threshold = 0.3
        valid = scores > threshold
        boxes, scores = boxes[valid], scores[valid]

        # Prepare JSON output
        results = [
            {"box": box.tolist(), "score": float(score)}
            for box, score in zip(boxes, scores)
        ]

# -----------------------------
# 7️⃣ Save JSON output
# -----------------------------
with open("detection_output.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Detection results saved to detection_output.json ({len(results)} boxes)")

# -----------------------------
# 8️⃣ Visualize detections
# -----------------------------
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(images_np[0])
for box in boxes:
    x_min, y_min, x_max, y_max = box
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='lime',
        facecolor='none'
    )
    ax.add_patch(rect)

plt.axis('off')
plt.tight_layout()
plt.savefig("detection_visualization.png", bbox_inches="tight")
print("✅ Visualization saved as detection_visualization.png")
plt.show()