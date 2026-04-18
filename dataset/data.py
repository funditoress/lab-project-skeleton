from torchvision.datasets import ImageFolder
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Veriyi 'data/' klasöründen okuyacak şekilde ayarladık
train_dataset = ImageFolder(root='data/tiny-imagenet-200/train', transform=transform)
val_dataset = ImageFolder(root='data/tiny-imagenet-200/val', transform=transform)