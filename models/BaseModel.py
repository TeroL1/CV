import torch
from torch import nn
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 0), # 3 32 32 -> 16 30 30
            nn.BatchNorm2d(num_features = 16),
            nn.GELU(),
            nn.Dropout2d(p = 0.3),

            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 0), # 16 30 30 -> 32 28 28
            nn.BatchNorm2d(num_features = 32),
            nn.GELU(),
            nn.Dropout2d(p = 0.4),

            nn.MaxPool2d(kernel_size = 2, stride = 2), # 32 28 28 -> 32 14 14
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0), # 32 14 14 -> 64 12 12
            nn.BatchNorm2d(num_features = 64),
            nn.GELU(),
            nn.Dropout2d(p = 0.2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0), # 64 12 12 -> 128 10 10
            nn.BatchNorm2d(num_features = 128),
            nn.GELU(),
            nn.Dropout2d(p = 0.3),

            nn.MaxPool2d(kernel_size = 2, stride = 2), # 128 10 10 -> 128 5 5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 128 * 5 * 5, out_features = 1600),
            nn.BatchNorm1d(num_features = 1600),
            nn.GELU(),
            nn.Dropout1d(p = 0.3),

            nn.Linear(in_features = 1600, out_features = 600),
            nn.BatchNorm1d(num_features = 600),
            nn.GELU(),
            nn.Dropout1d(p = 0.2),

            nn.Linear(in_features = 600, out_features = 200),
            nn.BatchNorm1d(num_features = 200),
            nn.GELU(),

            nn.Linear(in_features = 200, out_features = 10)
        )

        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        return self.classifier(self.feature(x))
    
    def train_model(self, train_data_loader, device, val_data_loader = None, num_epoches = 15, lr = 10**(-3), optimizer_func = torch.optim.AdamW, loss_func = torch.nn.CrossEntropyLoss()):
        net = self
        net.to(device)

        loss = loss_func
        optimizer = optimizer_func(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 0)

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