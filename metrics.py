import torch

def calculate_f1_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calculates F1 Score (Dice Coefficient) for binary segmentation.
    
    Args:
        preds (torch.Tensor): Model output logits (B, 1, H, W)
        targets (torch.Tensor): Ground truth masks (B, 1, H, W)
        threshold (float): Threshold to convert logits to binary predictions
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: F1 Score
    """
    # Apply sigmoid to convert logits to probabilities
    preds = torch.sigmoid(preds)
    
    # Binarize predictions
    preds = (preds > threshold).float()
    
    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # True Positives, False Positives, False Negatives
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    # F1 Score = 2*TP / (2*TP + FP + FN)
    f1 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    return f1.item()
