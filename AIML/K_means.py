from sklearn.cluster import KMeans
import cv2
import numpy as np
from AI_part import *
def Model_KMeans(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))  # Flatten

    kmeans = KMeans(n_clusters=k, random_state=0).fit(img)
    dominant_color = kmeans.cluster_centers_[0].astype(int)  # Most common cluster
    return dominant_color
