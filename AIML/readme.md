# Dominant Color Classification in Pokémon

## Overview
This project aims to determine the **dominant color** in Pokémon images using both **Machine Learning** and **Deep Learning** models. The classification is conducted using various techniques to ensure accurate color predictions.

## Models Used

### Machine Learning Models

1. **K-Means Clustering**
   - K-Means is an unsupervised learning algorithm used for clustering similar data points.
   - The algorithm groups pixels in an image into `k` clusters based on their color similarity using Euclidean distance.
   - The dominant color is determined by selecting the centroid of the most populated cluster.
   - K-Means is efficient for color quantization but may struggle with highly detailed or multi-colored images.

2. **Random Forest (RFT)**
   - Random Forest is a supervised learning algorithm that constructs multiple decision trees and combines their outputs to improve classification accuracy.
   - The model is trained on a dataset of Pokémon images with labeled dominant colors, learning to map pixel values to predefined color categories.
   - Each tree in the forest makes an independent prediction, and the final output is determined by majority voting.
   - Random Forest is robust against overfitting and generalizes well to unseen images but requires well-engineered features for optimal performance.

### Deep Learning Models

1. **Convolutional Neural Network (CNN)**
   - CNNs are deep learning models designed for image processing tasks by extracting hierarchical features from images.
   - The model consists of convolutional layers that apply filters to detect edges, textures, and color regions.
   - Pooling layers reduce the spatial dimensions while retaining essential features.
   - Fully connected layers map the extracted features to dominant color labels.
   - CNNs excel in recognizing complex patterns in images, making them effective for dominant color classification in Pokémon.

2. **Vision Transformer (ViT)**
   - ViT is a transformer-based deep learning model that applies self-attention mechanisms to image patches.
   - Unlike CNNs, which process local features, ViT divides an image into fixed-size patches and embeds them as tokenized vectors.
   - The model captures global contextual information, making it effective in recognizing overall color distributions rather than pixel-level patterns.
   - ViT has shown state-of-the-art performance in image classification but requires substantial computational resources for training.

## Methodology
- The Pokémon images are preprocessed and passed through different models.
- Machine Learning models (K-Means and RFT) analyze color distribution and classify the dominant color.
- Deep Learning models (CNN and ViT) are trained on Pokémon images to predict the most representative color.
- The results are compared to determine which model provides the best accuracy.

## Usage
Run the models on Pokémon images stored in the `cleaned/` directory to obtain dominant color predictions.

## Results
This project evaluates the effectiveness of Machine Learning and Deep Learning in Pokémon color classification. 

## Future Enhancements
- Improve model accuracy with better feature extraction.
- Train CNN and ViT on a larger dataset for better generalization.
- Experiment with additional deep learning architectures.

## Contributions
Contributions to this project are welcome.
