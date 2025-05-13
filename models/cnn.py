import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)  # ~336 (for RGB) or 120 (grayscale)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)           # ~2.6k

        # Feature map size after 2x2 pooling twice
        # 28x28 -> 7x7 for MNIST, 32x32 -> 8x8 for CIFAR
        self.feature_size = 32 * (7 * 7 if in_channels == 1 else 8 * 8)    # 1176 or 1536

        self.fc1 = nn.Linear(self.feature_size, 32)                        # 37.6k (MNIST) or 49.1k (CIFAR)
        self.fc2 = nn.Linear(32, num_classes)                              # 330

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
