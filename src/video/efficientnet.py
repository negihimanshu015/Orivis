import torch
import torch.nn as nn
import timm

class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone for deepfake detection with frame-level aggregation.
    """
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetBackbone, self).__init__()
                                    
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        
                                
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, NumFrames, C, H, W)
        """
        batch_size, num_frames, c, h, w = x.shape
                                                 
        x = x.view(-1, c, h, w)
        
                            
        features = self.backbone(x)                                
        features = self.pooling(features).flatten(1)                  
        
                                                                    
        features = features.view(batch_size, num_frames, -1)
        video_features = torch.mean(features, dim=1)                
        
                        
        logits = self.classifier(video_features)
        return logits

def get_efficientnet_model(model_name='efficientnet_b0', num_classes=2, pretrained=True):
    return EfficientNetBackbone(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
