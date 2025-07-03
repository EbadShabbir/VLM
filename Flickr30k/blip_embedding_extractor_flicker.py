import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipModel
import torch

# Set new paths
image_dir = "Images_dir"
label_file = "captions.txt"
output_embeddings = "blip_embeddings.npy"
output_labels = "blip_labels.npy"

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to load labels from the new label file
def load_labels(label_file):
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                image_path, caption = parts
                image_path = image_path.strip().strip('"')
                caption = caption.strip().strip('"')
                if image_path in label_dict:
                    label_dict[image_path].append(caption)
                else:
                    label_dict[image_path] = [caption]
    return label_dict

# Load labels
label_data = load_labels(label_file)

embeddings = []
labels = []

# Iterate over all images in the label data
for image_name, captions in label_data.items():
    image_path = os.path.join(image_dir, image_name)
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        continue
    
    try:
        # Select the first caption as the label (as an example)
        most_common_label = captions[0]
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Process the image with BLIP using the selected label as the text input
        inputs = processor(text=[most_common_label], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image.cpu().numpy()
        
        # Print the final label and tensor
        print(f"Image: {image_name}, Final Label: {most_common_label}")
        print(f"Embedding Tensor: {logits_per_image}")
        
        # Save the embeddings and label
        embeddings.append(logits_per_image)
        labels.append(most_common_label)
    
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
        continue

# Save embeddings and labels to .npy files
embeddings = np.concatenate(embeddings, axis=0)
np.save(output_embeddings, embeddings)
np.save(output_labels, np.array(labels))

print("Feature extraction and label processing completed.")
