
import os
import numpy as np
import torch
from transformers import BlipProcessor, BlipModel
from PIL import Image
import json

no_label_count = 0

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# File paths and directories (update these paths based on your CLEVR dataset location)
IMAGE_DIR = '/path/to/clever/dataset/images'
ANNOTATION_FILE = '/path/to/clever/dataset/annotations.json'
OUTPUT_PATH = '/path/to/output/directory'

# Load CLEVR annotations
with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)

# Extract image information and related questions
image_annotations = {}
for question in annotations['questions']:
    image_filename = question['image_filename']
    if image_filename not in image_annotations:
        image_annotations[image_filename] = []
    image_annotations[image_filename].append(question['question'])

image_ids = list(image_annotations.keys())

# Load the pretrained BLIP Processor and BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
model = BlipModel.from_pretrained("Salesforce/blip-itm-large-coco")

# Move model to GPU if available
model.to(device)


def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        # Move inputs to GPU if available
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            features = outputs.squeeze().cpu().numpy()

        return features

    except Exception as e:
        print(f"Error extracting features for image {image_path}: {str(e)}")
        return np.array([])  # Return empty array on error


# Feature extraction loop
features_list = []
labels_list = []
images_without_labels = []

# Check if there are any previously saved features and labels
if os.path.exists(os.path.join(OUTPUT_PATH, "checkpoint.npy")):
    checkpoint = np.load(os.path.join(OUTPUT_PATH, "checkpoint.npy"), allow_pickle=True).item()
    start_index = checkpoint['last_processed_index']
    features_list = checkpoint['features_list']
    labels_list = checkpoint['labels_list']
    no_label_count = checkpoint['no_label_count']
    images_without_labels = checkpoint['images_without_labels']
    print(f"Checkpoint found. Resuming from index {start_index}.")

else:
    start_index = 0

for i, img_id in enumerate(image_ids[start_index:], start=start_index):
    image_path = os.path.join(IMAGE_DIR, img_id)
    features = extract_features(image_path)

    if features.size == 0:
        print(f"Empty features for image: {image_path}")
        continue

    # Get annotations (questions) for the image
    questions = image_annotations[img_id]
    if not questions:
        no_label_count += 1
        images_without_labels.append(image_path)
        continue

    # Create a data point for each question
    for question in questions:
        features_list.append(features)
        labels_list.append(question)

    if (i + 1) % 11800 == 0:
        print(f"{i + 1} images processed")

    # Save checkpoint every 500 images processed
    if (i + 1) % 500 == 0:
        checkpoint = {
            'last_processed_index': i + 1,
            'features_list': features_list,
            'labels_list': labels_list,
            'no_label_count': no_label_count,
            'images_without_labels': images_without_labels
        }
        np.save(os.path.join(OUTPUT_PATH, "checkpoint.npy"), checkpoint)

# Convert features_list and labels_list to arrays
train_features_array = np.array(features_list)

# Since questions are strings, we'll keep them as a list of strings
train_labels_array = np.array(labels_list, dtype=object)

# Ensure train_features_array and train_labels_array are correctly shaped
print("Features array shape:", train_features_array.shape)
print("Labels array shape:", train_labels_array.shape)

# Save final features and labels
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Ensure OUTPUT_PATH exists
np.save(os.path.join(OUTPUT_PATH, "blip_train_features.npy"), train_features_array)
np.save(os.path.join(OUTPUT_PATH, "blip_train_labels.npy"), train_labels_array)

# Print images without labels
print("Total Images without labels: ", no_label_count)
print("Images without labels: ", images_without_labels)

# Remove checkpoint file if processing completed successfully
if os.path.exists(os.path.join(OUTPUT_PATH, "checkpoint.npy")):
    os.remove(os.path.join(OUTPUT_PATH, "checkpoint.npy"))
    print("Removed checkpoint file.")
