import torch
import torch.nn as nn
from torchvision import models
class DamageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DamageClassifier, self).__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)