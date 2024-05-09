from tqdm import tqdm
import torch
import pdb
import torch.nn.functional as F
from itertools import chain
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# batch_size = 4
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# conv1x1 = torch.nn.Conv2d(12, 1, kernel_size=1).to(device)
# linear = torch.nn.Sequential(torch.nn.Linear(768, 256),
#                              torch.nn.ReLU(),
#                              torch.nn.Linear(256, 128),
#                              torch.nn.ReLU(),
#                              torch.nn.Linear(128, 64),
#                              torch.nn.ReLU()).to(device)

# cls_head = torch.nn.Linear(161472, 5).to(device)
def train(epoch, model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    gts = []
    for inputs, targets in tqdm(train_loader, desc="Training"):
        
        inputs, targets = inputs.to(device).unsqueeze(1).float(), targets.to(device).float()
        targets = F.one_hot(targets.long(), num_classes=5)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs.softmax(dim=1), targets.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.argmax(outputs.softmax(dim=1), dim=1)
        predictions.append(predicted.detach().cpu())
        gts.append(torch.argmax(targets, dim=1).detach().cpu())
        
        total += targets.size(0)
        correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()

    predictions = np.array(list(chain(*predictions)))
    gts = np.array(list(chain(*gts)))
    
    df = pd.DataFrame({'predictions': predictions, 'gts':gts})
    sns.stripplot(x='gts', y='predictions', data=df, size=5, linewidth=0.5, hue='predictions')
    plt.show()
    plt.savefig(f'{epoch}_train.png')
    plt.close()

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc

def validate(epoch, model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    gts = []
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device).unsqueeze(1).float(), targets.to(device).float()
            targets = F.one_hot(targets.long(), num_classes=5)
            outputs = model(inputs)
            loss = criterion(outputs.softmax(dim=1), targets.float())

            running_loss += loss.item()
            predicted = torch.argmax(outputs.softmax(dim=1), dim=1)
            
            predictions.append(predicted.detach().cpu())
            gts.append(torch.argmax(targets, dim=1).detach().cpu())
        
            total += targets.size(0)
            correct += predicted.eq(torch.argmax(targets, dim=1)).sum().item()

    predictions = np.array(list(chain(*predictions)))
    gts = np.array(list(chain(*gts)))
    df = pd.DataFrame({'predictions': predictions, 'gts':gts})
    sns.stripplot(x='gts', y='predictions', data=df, size=5, linewidth=0.5, hue='predictions')
    plt.show()
    plt.savefig(f'{epoch}_test.png')
    plt.close()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc