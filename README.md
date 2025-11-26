A deep learning project for automatic classification of brain tumors from MRI images into three categories: **Meningioma**, **Glioma**, and **Pituitary tumors** using Convolutional Neural Networks (CNN).

# Project Overview

This project implements a complete deep learning pipeline for medical image analysis, demonstrating the application of CNNs to classify brain tumors with high accuracy. The model processes raw MRI scans through preprocessing, feature extraction, and classification stages.

# Dataset
- Source: Brain Tumor MRI Dataset (.mat files)
- Classes: 3 (Meningioma, Glioma, Pituitary Tumor)
- Samples: 3,064 MRI images

# Model Architecture
The CNN architecture consists of:
- Input: 256×256 grayscale MRI images
- Convolutional Blocks: 3 blocks with Batch Normalization and MaxPooling
- Feature Maps: 32 → 64 → 128 filters
- Classifier: Dense layers with Dropout regularization
- Output: 3-class softmax classification
