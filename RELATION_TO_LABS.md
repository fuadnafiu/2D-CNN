# Project Relation to Course Labs (EEE385L)

This document outlines how the **Hurricane Damage Classification** project applies, combines, and extends the concepts learned in **Lab 6** and **Lab 7**.

## Overview
This project serves as an advanced application of the coursework. While Lab 6 introduced the fundamentals of neural networks and Lab 7 introduced image processing with CNNs, this project applies those concepts to a complex, real-world problem using state-of-the-art techniques (Transfer Learning).

---

## 1. Relation to Lab 6: Neural Networks
**Lab 6 Topic:** Training a Neural Network with Keras (Dense Layers) for Regression.

### Concepts Applied:
*   **Fully Connected (Dense) Layers:**
    *   **In Lab 6:** You used `Dense` layers to process tabular data features.
    *   **In This Project:** We use Fully Connected layers (defined as `nn.Linear` in PyTorch) at the very end of our model. After the CNN extracts features from the image, these dense layers make the final decision (Damage vs No Damage).
    *   *See `model.py`:* `self.backbone.fc = nn.Sequential(...)`

*   **Activation Functions:**
    *   **In Lab 6:** You likely used `ReLU` for hidden layers.
    *   **In This Project:** We use `ReLU` in our classifier head to introduce non-linearity.

*   **Optimization:**
    *   **In Lab 6:** You used an optimizer (like Adam or SGD) to minimize loss.
    *   **In This Project:** We use `optim.Adam` in `train.py` to update our model weights.

### Key Differences:
*   **Task:** Lab 6 was *Regression* (predicting a continuous number). This project is *Classification* (predicting a category).
*   **Loss Function:** Lab 6 used Mean Squared Error (MSE). This project uses Cross Entropy Loss (`nn.CrossEntropyLoss`), which is standard for classification.

---

## 2. Relation to Lab 7: Convolutional Neural Networks (CNN)
**Lab 7 Topic:** Classification using CNNs (MNIST Dataset).

### Concepts Applied:
*   **Convolutional Layers:**
    *   **In Lab 7:** You built a network with `Conv2D` layers to detect simple patterns (edges, curves) in handwritten digits.
    *   **In This Project:** We use **ResNet34**, which is a deep stack of Convolutional layers. It works exactly the same way but is much deeper, allowing it to recognize complex textures (rubble, broken roofs, flooding) instead of just simple lines.

*   **Pooling Layers:**
    *   **In Lab 7:** You used `MaxPooling` to reduce image size.
    *   **In This Project:** ResNet34 uses pooling layers internally to downsample the image as it processes it.

*   **Image Preprocessing:**
    *   **In Lab 7:** You normalized pixel values (0-255 to 0-1).
    *   **In This Project:** We perform normalization in `dataset.py` using `A.Normalize`, but we also add **Data Augmentation** (rotation, flipping) to make the model more robust.

---

## 3. How This Project Extends the Labs
This project goes beyond the basic labs in two major ways, representing a "Final Project" level of complexity:

1.  **Transfer Learning:**
    *   In Lab 7, you trained a CNN from scratch.
    *   In this project, we use **Transfer Learning**. We start with a ResNet34 model that has *already* been trained on 1.2 million images (ImageNet). This allows us to achieve 99% accuracy with only 10,000 images, which would be impossible if we trained from scratch like in Lab 7.

2.  **Real-World Data:**
    *   Lab 7 used MNIST (28x28 grayscale images, perfectly centered).
    *   This project uses Satellite Imagery (128x128 color images, noisy, varying angles). This requires more robust data loading (`dataset.py`) and preprocessing.

## Summary Table

| Feature | Lab 6 | Lab 7 | This Project |
| :--- | :--- | :--- | :--- |
| **Input Data** | Tabular (CSV) | Images (28x28 Grayscale) | Images (128x128 RGB) |
| **Core Architecture** | Dense Neural Network | Simple CNN | **ResNet34 (Deep CNN)** |
| **Task Type** | Regression | Classification | **Classification** |
| **Training Method** | From Scratch | From Scratch | **Transfer Learning** |
| **Framework** | Keras/TensorFlow | Keras/TensorFlow | **PyTorch** |
