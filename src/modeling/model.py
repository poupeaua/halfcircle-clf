"""
NN model class
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
import torch.nn.functional as F

class HCCLF(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjusted for 128x128 input size
        self.fc2 = nn.Linear(256, 1)  # Output a single value

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (128*128 -> 64*64)
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))  # (64*64 -> 32*32)
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))  # (32*32 -> 16*16)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Output between 0 and 1
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1)
        outputs = self.forward(images)
        loss = F.binary_cross_entropy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
