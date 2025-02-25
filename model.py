import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_way):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 84 -> 42
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 42 -> 21
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 21 -> 10
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 10 -> 5
        )
        self.classifier = nn.Linear(64 * 5 * 5, n_way)
        
    def forward(self, x):
        x = x.view(-1, 3, 84, 84)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return self.classifier(x)