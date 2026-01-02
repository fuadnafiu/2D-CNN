import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from dataset import HurricaneDamageDataset, get_transforms
from model import DamageClassifier

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10
IMAGE_SIZE = 128 
DATA_ROOT = "hurricane_data"

# ==========================================
# 2. TRAINING HELPER
# ==========================================
def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        # Metrics
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(loader, model, loss_fn):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. Prepare Data
    # Check if data exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data directory '{DATA_ROOT}' not found. Please unzip archive.zip first.")
        return

    train_damage = glob.glob(os.path.join(DATA_ROOT, "train_another", "damage", "*.jpeg"))
    train_no_damage = glob.glob(os.path.join(DATA_ROOT, "train_another", "no_damage", "*.jpeg"))
    
    val_damage = glob.glob(os.path.join(DATA_ROOT, "validation_another", "damage", "*.jpeg"))
    val_no_damage = glob.glob(os.path.join(DATA_ROOT, "validation_another", "no_damage", "*.jpeg"))

    # Combine lists
    train_files = train_damage + train_no_damage
    # 1 = Damage, 0 = No Damage
    train_labels = [1]*len(train_damage) + [0]*len(train_no_damage)
    
    val_files = val_damage + val_no_damage
    val_labels = [1]*len(val_damage) + [0]*len(val_no_damage)

    print(f"Training Samples: {len(train_files)} (Damage: {len(train_damage)}, No Damage: {len(train_no_damage)})")
    print(f"Validation Samples: {len(val_files)}")

    if len(train_files) == 0:
        print("Error: No training files found. Check the folder structure inside 'hurricane_data'.")
        return

    # Create Datasets
    train_ds = HurricaneDamageDataset(train_files, train_labels, transforms=get_transforms('train', IMAGE_SIZE))
    val_ds = HurricaneDamageDataset(val_files, val_labels, transforms=get_transforms('val', IMAGE_SIZE))

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Setup Model
    model = DamageClassifier(num_classes=2).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE=="cuda"))

    # 3. Training Loop
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, val_acc, val_f1 = validate(val_loader, model, loss_fn)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")

if __name__ == "__main__":
    main()
