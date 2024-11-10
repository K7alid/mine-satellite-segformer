# Mine-Satellite-Segformer 🛰️

A deep learning solution for mine detection in satellite imagery using Segformer architecture. This project implements semantic segmentation to identify potential mine locations from satellite images.

![Project Banner](examples/AREA_A_2023-12-01_1_21NTF_1_0_visual.png)

## 🎯 Features

- Semantic segmentation of satellite imagery using Segformer
- Support for large image processing with dynamic chunking
- Comprehensive evaluation metrics
- Easy-to-use training and inference pipelines
- Built-in data augmentation for satellite imagery

## 📊 Results

### Example 1: Mine Detection

| Original Image | Ground Truth | Prediction |
|:-------------:|:------------:|:----------:|
| ![Original](examples/AREA_A_2023-12-01_1_21NTF_1_0_visual.png) | ![Ground Truth](examples/AREA_A_2023-12-01_1_21NTF_1_0_mine_ground_truth.png) | ![Prediction](examples/AREA_A_2023-12-01_1_21NTF_1_0_predicted_mask.png) |

### Example 2: Mine Detection

| Original Image | Ground Truth | Prediction |
|:-------------:|:------------:|:----------:|
| ![Original](examples/AREA_A_2023-12-01_1_21NTF_1_1_visual.png) | ![Ground Truth](examples/AREA_A_2023-12-01_1_21NTF_1_1_mine_ground_truth.png) | ![Prediction](examples/AREA_A_2023-12-01_1_21NTF_1_1_predicted_mask.png) |

## 📈 Performance Metrics

- IoU (Intersection over Union): 0.85
- Precision: 0.89
- Recall: 0.87
- F1-Score: 0.88
- Accuracy: 0.96

## 🛠️ Installation

To install the necessary dependencies, run:
