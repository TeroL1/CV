import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

from types import MethodType

import numpy as np

from tqdm import tqdm

def train_model(self, train_data_loader, device, val_data_loader = None, num_epoches = 15, lr = 10**(-3), optimizer_func = torch.optim.AdamW, loss_func = torch.nn.CrossEntropyLoss()):
    net = self
    net.to(device)

    loss = loss_func
    optimizer = optimizer_func(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epoches, eta_min = 0)

    for epoch in tqdm(range(num_epoches)):
        epoch_losses = []

        for X_batch, y_batch in train_data_loader:
            optimizer.zero_grad()
            net.train()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = net.forward(X_batch)
                
            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

            epoch_losses.append(loss_value.detach().item()) 

        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.train_loss.append(epoch_avg_loss)
        print(f'Train - {epoch}:{epoch_avg_loss}')

        if val_data_loader is not None:
            self.eval()  
            epoch_val_losses = []
                
            with torch.no_grad(): 
                for X_val, y_val in val_data_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = self(X_val)
                    val_loss = loss_func(val_outputs, y_val)
                    epoch_val_losses.append(val_loss.detach().cpu().item())
                
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            self.val_loss.append(avg_val_loss)
            print(f'Val - {epoch}:{avg_val_loss}')

        scheduler.step()
            
def plot_loss(self):
    plt.figure(figsize=(12, 8)) 
    plt.plot(range(len(self.train_loss)), self.train_loss, label = "Train")
    if self.val_loss: 
        plt.plot(range(len(self.val_loss)), self.val_loss, label = "Val")
    plt.title("Ошибка во время обучения")
    plt.ylabel("Ошибка")
    plt.xlabel("Эпоха")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


class CustomResNet18:
    def __init__(self):
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.model.fc = nn.Sequential(
            nn.Linear(512, 380),
            nn.BatchNorm1d(380),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(380, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(160, 65),
            nn.BatchNorm1d(65),
            nn.ReLU(),
            
            nn.Linear(65, 10)
        )

        for name, param in self.model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model.train_loss = []
        self.model.val_loss = []
        self.model.train_model = MethodType(train_model, self.model)
        self.model.plot_loss = MethodType(plot_loss, self.model)

    def get_model(self):
        return self.model