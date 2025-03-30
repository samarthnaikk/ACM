import cv2
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision import transforms

model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.eval()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return transform(image).unsqueeze(0)

def extract_vit_features(image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)  
    return features.squeeze().cpu().numpy() 

# Find Dominant Colors using K-Means
def find_dominant_colors(image, features, k=5):
    pixels = image.reshape(-1, 3)  

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_
    return dominant_colors.astype(int)

def Model_ViT(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = preprocess_image(image_path)
    features = extract_vit_features(image_tensor)
    dominant_colors = find_dominant_colors(image, features, k=5)
    for color in dominant_colors:
        if color[0] + color[1] + color[2] >= 12: #Threshold of [4 4 4] for color dominance
            return color
    return dominant_colors[0]
