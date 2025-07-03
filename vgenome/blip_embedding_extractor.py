
import os
import json
import numpy as np
from collections import Counter
from PIL import Image
from transformers import AutoProcessor, BlipModel
import torch

# Set paths
image_dir = "img"
label_file = "objects.json"
output_embeddings = "blip_embeddings.npy"
output_labels = "blip_labels.npy"

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load labels
with open(label_file, 'r') as f:
    label_data = json.load(f)

# Function to extract the most frequent label for an image
def get_most_frequent_label(objects):
    all_labels = []
    for obj in objects:
        all_labels.extend(obj.get('names', []))
    if not all_labels:
        return None
    most_common_label = Counter(all_labels).most_common(1)[0][0]
    return most_common_label

embeddings = []
labels = []

# Iterate over all images
for image_info in label_data:
    image_id = image_info['image_id']
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        continue
    
    try:
        # Get the most frequent label
        most_common_label = get_most_frequent_label(image_info['objects'])
        if most_common_label is None:
            print(f"No valid labels for image {image_id}.")
            continue
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Process the image with BLIP using the most frequent label as the text input
        inputs = processor(text=[most_common_label], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image.cpu().numpy()
        
        # Print the final label and tensor
        print(f"Image ID: {image_id}, Final Label: {most_common_label}")
        print(f"Embedding Tensor: {logits_per_image}")
        
        # Save the embeddings and label
        embeddings.append(logits_per_image)
        labels.append(most_common_label)
    
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        continue

# Save embeddings and labels to .npy files
embeddings = np.concatenate(embeddings, axis=0)
np.save(output_embeddings, embeddings)
np.save(output_labels, np.array(labels))

print("Feature extraction and label processing completed.")
