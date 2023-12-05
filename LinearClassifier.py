import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        return self.fc(x)