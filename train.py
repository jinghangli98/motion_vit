from dataset import train_loader, test_loader
import torch
from model import model
import torch.optim as optim
import torch.nn as nn
from utils import train, validate
from torch.cuda.amp import GradScaler, autocast


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('/ix1/tibrahim/jil202/11-motion-correction/motion_vit/checkpoint/weight_9.pth'))
model.to(device)

# scaler = GradScaler()
best_loss = 1.4
for epoch in range(100):
    train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(epoch, model, test_loader, criterion, device)
    
    print(f'Epoch: {epoch}, Accuracy: {train_acc:.4f} (Train) {val_acc:.4f} (Test), Loss: {train_loss:.4f} (Train) {val_loss:.4f} (Test)')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'./checkpoint/weight_{epoch}.pth')