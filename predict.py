import torch
import cv2
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
import random
from model import DamageClassifier
MODEL_PATH = 'best_model.pth'
TEST_DIR = 'hurricane_data/test_another'
IMAGE_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_model(model_path):
    model = DamageClassifier()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
def get_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
def predict_image(model, image_path, transforms):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transforms(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE) 
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
    prob_damage = probabilities[0][1].item()
    prob_no_damage = probabilities[0][0].item()
    prediction = "Damage" if predicted_class.item() == 1 else "No Damage"
    confidence = prob_damage if prediction == "Damage" else prob_no_damage
    return prediction, confidence
def main():
    parser = argparse.ArgumentParser(description='Predict damage on hurricane images')
    parser.add_argument('--image', type=str, help='Path to a single image to predict')
    args = parser.parse_args()
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    transforms = get_transforms()
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        print(f"\nPredicting on single image: {args.image}")
        result = predict_image(model, args.image, transforms)
        if result:
            prediction, confidence = result
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
        return
    damage_images = glob.glob(os.path.join(TEST_DIR, 'damage', '*.*'))
    no_damage_images = glob.glob(os.path.join(TEST_DIR, 'no_damage', '*.*'))
    if not damage_images and not no_damage_images:
        print(f"No images found in {TEST_DIR}")
        return
    samples = []
    if damage_images:
        samples.extend(random.sample(damage_images, min(3, len(damage_images))))
    if no_damage_images:
        samples.extend(random.sample(no_damage_images, min(3, len(no_damage_images))))
    print(f"\nRunning predictions on {len(samples)} random images from {TEST_DIR}...\n")
    print(f"{'Image Path':<60} | {'True Label':<10} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 100)
    correct = 0
    for img_path in samples:
        true_label = "Damage" if "damage" in os.path.basename(os.path.dirname(img_path)) and "no_damage" not in os.path.basename(os.path.dirname(img_path)) else "No Damage"
        parent_folder = os.path.basename(os.path.dirname(img_path))
        if parent_folder == 'damage':
            true_label = "Damage"
        elif parent_folder == 'no_damage':
            true_label = "No Damage"
        result = predict_image(model, img_path, transforms)
        if result:
            prediction, confidence = result
            if prediction == true_label:
                correct += 1
            display_path = "..." + img_path[-55:] if len(img_path) > 55 else img_path
            print(f"{display_path:<60} | {true_label:<10} | {prediction:<10} | {confidence:.4f}")
    print("-" * 100)
    print(f"Accuracy on these samples: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
if __name__ == "__main__":
    main()