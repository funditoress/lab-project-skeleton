import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        self.pool = nn.MaxPool2d(2, 2) # Boyutu küçültmek için şart
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Kanalları 256'ya çıkarmak için
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Boyutu 1x1 yapıp tam 256 sayı elde etmek için

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224

        x = self.pool(x)           # Boyut: 112x112
        x = self.conv2(x).relu()   # Kanal: 128
        x = self.pool(x)           # Boyut: 56x56
        x = self.conv3(x).relu()   # Kanal: 256
        x = self.avgpool(x)        # Boyut: 1x1 (Toplam 256 değer kaldı)

        x = torch.flatten(x, 1)    # Matrisi düzleştir (Linear katman için)
        x = self.fc1(x)            # Tahmin: 200 sınıf


        return x