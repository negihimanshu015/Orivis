import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAudioCNN(nn.Module):
    """
    A simple CNN classifier for log-mel spectrograms.
    Outputs raw logits for BCEWithLogitsLoss.
    """
    def __init__(self, num_classes: int = 1):
        super(SimpleAudioCNN, self).__init__()
        
                      
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
                      
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
                         
                                 
                              
                                    
                                    
                                                  
                                                       
        self.fc1 = nn.Linear(32 * 32 * 100, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
                                 
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
                 
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        
        return logits.squeeze(1) if logits.shape[1] == 1 else logits
