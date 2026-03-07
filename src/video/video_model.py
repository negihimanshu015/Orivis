import torch
import torch.nn as nn
import timm

class XceptionBaseline(nn.Module):
    """
    Xception model for face forgery detection.
    Pretrained on ImageNet by default.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(XceptionBaseline, self).__init__()
                                
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        
                                                   
                                                      
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, C, H, W) for single frame 
        OR (Batch, NumFrames, C, H, W) if using aggregator.
        For baseline, we assume frames are passed as a batch.
        """
        if len(x.shape) == 5:
                                                                                
            batch_size, num_frames, c, h, w = x.shape
            x = x.view(-1, c, h, w)
            logits = self.backbone(x)
                                                                                           
            logits = logits.view(batch_size, num_frames, -1)
            logits = torch.mean(logits, dim=1)
            return logits
            
        return self.backbone(x)

def get_xception_model(num_classes=2, pretrained=True):
    return XceptionBaseline(num_classes=num_classes, pretrained=pretrained)
