import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def find_top_dominant_colors(image, k=5):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(-counts)
    dominant_colors = kmeans.cluster_centers_[sorted_indices].astype(int)
    return dominant_colors, counts[sorted_indices]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def extract_vit_features(image_tensor):
    return np.mean(image_tensor, axis=(0, 1))

def find_dominant_colors(image, features, k=5):
    dominant_colors, counts = find_top_dominant_colors(image, k)
    return dominant_colors

def Model_CNN(image_path):
    image = load_image(image_path)
    image_tensor = preprocess_image(image_path)
    features = extract_vit_features(image_tensor)
    dominant_colors = find_dominant_colors(image, features, k=5)
    for color in dominant_colors:
        if np.sum(color) >= 12:
            return color
    return dominant_colors[0]
