import torch
import torch.nn as nn
import torch.optim as optim # Optimizasyon için şart
import wandb
from dataset.data import train_dataset, val_dataset # Senin dosyan
from models.model import CustomNet                # Senin dosyan

if __name__ == '__main__':
    # Cihaz ayarı
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 1. DataLoader'ları oluştur
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Modeli başlat
    model = CustomNet().to(device)

    # 3. Wandb başlat [cite: 42]
    wandb.init(project='mldl_projem')

    # 4. KRİTİK EKSİKLERİ BURAYA EKLEDİK:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0 # Bu değişkeni tanımlamazsan hata alırsın!

    # ... (Senin train ve validate fonksiyonların burada kalsın, aynen durabilir) ...
    def train(epoch, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # todo...
            # 1. Gradyanları sıfırla (Eski hataları temizle)
            optimizer.zero_grad()

            # 2. Tahmin yap (Forward pass)
            outputs = model(inputs)

            # 3. Hatayı hesapla (Loss)
            loss = criterion(outputs, targets)

            # 4. Geriye yayılım (Backpropagation - Hata tespiti)
            loss.backward()

            # 5. Ağırlıkları güncelle (Öğrenme adımı)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # Validation loop
    def validate(model, val_loader, criterion):
        model.eval()
        val_loss = 0

        correct, total = 0, 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()

                # todo...
                # 1. Tahmin yap (Forward pass)
                outputs = model(inputs)

                # 2. Hatayı hesapla (Loss)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total

        print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
        return val_accuracy

    # 5. Eğitim Döngüsü (Döngünün içine optimizer ve criterion'u gönderiyoruz)
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        
        best_acc = max(best_acc, val_accuracy)
        wandb.log({"val_accuracy": val_accuracy, "epoch": epoch}) # Wandb'ye gönder

    print(f'Best validation accuracy: {best_acc:.2f}%')
    torch.save(model.state_dict(), "best_model.pth")
    print("Model 'best_model.pth' olarak kaydedildi!")