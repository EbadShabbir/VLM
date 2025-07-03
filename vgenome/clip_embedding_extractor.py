
#CLIP

# from PIL import Image
# import requests
# from transformers import CLIPProcessor, CLIPModel
# import torch

# # Check if GPU is available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load the model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Load the image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # Process the image and text inputs
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# # Move the inputs to the device
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # Get the model outputs
# outputs = model(**inputs)

# # Extract the logits per image
# logits_per_image = outputs.logits_per_image

# # Print the logits tensor
# print("Logits per image:", logits_per_image)



#CLIP for VGenome

import os
import json
import numpy as np
from collections import Counter
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Set paths
image_dir = "img"
label_file = "objects.json"
output_embeddings = "clip_embeddings.npy"
output_labels = "labels.npy"

# Load the CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        
        # Process the image with CLIP using the most frequent label as the text input
        inputs = processor(text=[most_common_label], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds.cpu().numpy()
        
        # Print the final label and tensor
        print(f"Image ID: {image_id}, Final Label: {most_common_label}")
        print(f"Embedding Tensor: {image_embeds}")
        
        # Save the embeddings and label
        embeddings.append(image_embeds)
        labels.append(most_common_label)
    
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        continue

# Save embeddings and labels to .npy files
embeddings = np.concatenate(embeddings, axis=0)
np.save(output_embeddings, embeddings)
np.save(output_labels, np.array(labels))

print("Feature extraction and label processing completed.")





