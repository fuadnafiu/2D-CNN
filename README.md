# Hurricane Damage Classification Project

This project implements a binary image classifier to detect damage in satellite/aerial imagery of hurricane-affected areas. It uses a ResNet34 architecture pre-trained on ImageNet and fine-tuned for this specific task.

## Project Structure

### Documentation
*   **`RELATION_TO_LABS.md`**: Explains how this project applies and extends the concepts from **EEE385L Lab 6 and Lab 7**.

### Core Files
These are the main files used for the classification task:

*   **`train.py`**: The main training script.
    *   Loads the dataset from `hurricane_data/`.
    *   Initializes the `DamageClassifier` model.
    *   Trains the model for a specified number of epochs.
    *   Saves the best performing model to `best_model.pth`.
    *   Calculates Accuracy and F1 Score.

*   **`predict.py`**: The inference script.
    *   Loads the trained `best_model.pth`.
    *   Can predict on a random batch of test images.
    *   Can predict on a specific single image file using the `--image` argument.

*   **`model.py`**: Defines the neural network architecture.
    *   Class: `DamageClassifier`.
    *   Uses a ResNet34 backbone (from `torchvision`).
    *   Modifies the final fully connected layer for binary classification (Damage vs No Damage).

*   **`dataset.py`**: Handles data loading and preprocessing.
    *   Class: `HurricaneDamageDataset`.
    *   Function: `get_transforms()` defines image augmentations (Resize, Flip, Rotate, Normalize) using the `albumentations` library.

*   **`unzip_data.py`**: A utility script used to extract the initial dataset zip file.

### Data
*   **`hurricane_data/`**: The directory containing the image dataset.
    *   Structure includes `train_another`, `validation_another`, `test_another` folders.
    *   Each contains `damage` and `no_damage` subdirectories.

### Legacy / Unused Files
These files were part of a previous iteration (Segmentation task) or are scratchpads:
*   `siamese_unet.py`: (Legacy) A Siamese U-Net architecture for change detection/segmentation.
*   `process_data.py`: (Legacy) Script for processing tar files for segmentation.
*   `metrics.py`: (Legacy) Custom metric calculations for segmentation.
*   `test.py`: A scratchpad file for testing snippets.

## Code Explanation

This section provides a step-by-step explanation of the codebase.

### 1. `dataset.py` (Data Loading)
This file handles loading images from the disk and preparing them for the model.
*   **`HurricaneDamageDataset` Class**:
    *   Inherits from PyTorch's `Dataset` class.
    *   **`__init__`**: Stores the list of image paths and their corresponding labels (0 for No Damage, 1 for Damage).
    *   **`__getitem__`**:
        1.  Reads an image using OpenCV (`cv2.imread`).
        2.  Converts it from BGR to RGB color space.
        3.  Applies image transformations (augmentations).
        4.  Returns the processed image tensor and its label.
*   **`get_transforms` Function**:
    *   Defines the image preprocessing pipeline using the `albumentations` library.
    *   **Training Phase**: Resizes images to 128x128, applies random flips (horizontal/vertical), random rotation (up to 30 degrees), and random brightness/contrast adjustments. This "Data Augmentation" helps the model generalize better.
    *   **Validation/Test Phase**: Only resizes and normalizes the images (no random changes).
    *   **Normalization**: Standardizes pixel values using ImageNet mean and standard deviation.

### 2. `model.py` (Neural Network Architecture)
This file defines the structure of the deep learning model.
*   **`DamageClassifier` Class**:
    *   Inherits from `nn.Module`.
    *   **Backbone**: Uses **ResNet34**, a powerful Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset. This allows us to use "Transfer Learning".
    *   **Classifier Head**: We replace the original final layer of ResNet34 with a custom sequence:
        1.  `Linear(512, 256)`: Reduces features from 512 to 256.
        2.  `ReLU()`: Activation function to introduce non-linearity.
        3.  `Dropout(0.5)`: Randomly turns off 50% of neurons during training to prevent overfitting.
        4.  `Linear(256, 2)`: Final output layer with 2 neurons (one for "No Damage", one for "Damage").

### 3. `train.py` (Training Loop)
This is the main script to train the model.
*   **Configuration**: Sets hyperparameters like `BATCH_SIZE` (32), `LEARNING_RATE` (0.0001), and `NUM_EPOCHS` (10).
*   **Data Preparation**:
    *   Scans the `hurricane_data` folder to find all image files.
    *   Splits them into Training and Validation sets.
    *   Creates `DataLoader` objects to feed data to the model in batches.
*   **Training Process (`train_one_epoch`)**:
    *   Iterates through the training data.
    *   **Forward Pass**: Passes images through the model to get predictions.
    *   **Loss Calculation**: Compares predictions with actual labels using `CrossEntropyLoss`.
    *   **Backward Pass**: Calculates gradients and updates model weights using the `Adam` optimizer.
*   **Validation Process (`validate`)**:
    *   Evaluates the model on the validation set without updating weights.
    *   Calculates Accuracy and F1 Score.
*   **Model Saving**: Saves the model with the highest validation accuracy as `best_model.pth`.

### 4. `predict.py` (Inference)
This script uses the trained model to make predictions on new images.
*   **`load_model`**: Recreates the `DamageClassifier` architecture and loads the saved weights from `best_model.pth`.
*   **`predict_image`**:
    *   Loads a single image.
    *   Preprocesses it (Resize, Normalize, ToTensor).
    *   Passes it through the model.
    *   Applies `Softmax` to convert outputs into probabilities (confidence scores).
    *   Returns the predicted class ("Damage" or "No Damage") and the confidence score.
*   **`main`**:
    *   Can run in two modes:
        1.  **Single Image**: If `--image` argument is provided, it predicts for that specific file.
        2.  **Batch Test**: If no argument is provided, it picks random images from the test folder and prints predictions.

### 5. `unzip_data.py` (Setup Utility)
A simple helper script to extract the dataset.
*   Checks if `archive.zip` exists.
*   Extracts contents to the `hurricane_data` directory.
*   Handles errors like missing files or corrupted zips.

## Setup & Installation

1.  **Environment**: Ensure you are using Python 3.10+.
2.  **Dependencies**: Install the required packages:
    ```bash
    pip install torch torchvision opencv-python albumentations scikit-learn
    ```

## Usage

### 1. Training the Model
To train the model from scratch:
```bash
python train.py
```
This will create `best_model.pth` upon completion.

### 2. Running Predictions
**Test on random images from the dataset:**
```bash
python predict.py
```

**Test on a specific image file:**
```bash
python predict.py --image "path/to/your/image.jpg"
```
Example:
```bash
python predict.py --image "Hurricane-QA.jpg"
```

## Model Details
*   **Architecture**: ResNet34
*   **Input Size**: 128x128 pixels
*   **Classes**: 
    *   0: No Damage
    *   1: Damage
*   **Performance**: ~99% Accuracy on Validation Set.
