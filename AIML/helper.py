import cv2
import numpy as np
import os

def remove_background(image_path, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    
    file_name = os.path.basename(image_path)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, img)
    return output_path

input_folder = "Pokemon"
output_folder = "cleaned"

for file in os.listdir(input_folder):
    if file.endswith(".png"):
        input_path = os.path.join(input_folder, file)
        output_path = remove_background(input_path, output_folder)
        print(f"Saved: {output_path}")
