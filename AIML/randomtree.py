import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    
def find_top_dominant_colors_rf(image, k=5):
    pixels = image.reshape(-1, 3)
    pixels_rounded = (pixels // 10) * 10
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    labels = np.random.randint(0, k, size=len(pixels_rounded))
    rf.fit(pixels_rounded, labels)  
    predictions = rf.predict(pixels_rounded)
    cluster_counts = Counter(predictions)
    most_common_clusters = [c[0] for c in cluster_counts.most_common(k)]
    dominant_colors = []
    for cluster in most_common_clusters:
        cluster_pixels = pixels[predictions == cluster]
        dominant_colors.append(np.mean(cluster_pixels, axis=0))

    return np.array(dominant_colors, dtype=int)

def Model_RFT(image_path):
    image = load_image(image_path)

    dominant_colors = find_top_dominant_colors_rf(image, k=5)
    for color in dominant_colors:
        if color[0]+color[1]+color[2]>=75: #Threshold [25 25 25]
            return color
    return dominant_colors[0]
    
