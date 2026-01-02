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
