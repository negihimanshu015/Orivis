import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List

class GradCAM:
    """
    Implementation of Grad-CAM for generating saliency maps.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        def save_activation(module, input, output):
            self.activations = output
            
            # Use tensor hooks to safely catch gradients flowing through this activation
            def _store_grad(grad):
                self.gradients = grad
            
            if output.requires_grad:
                output.register_hook(_store_grad)
                
        self.handlers.append(self.target_layer.register_forward_hook(save_activation))

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate localized heatmap for a specific class.
        """
        self.model.eval()
        
        # Ensure input tracks history to force gradient propagation to the target layer
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

                                              
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
                                
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def visualize_on_image(self, img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        Overlay heatmap on the original image.
        """
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
                                    
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlayed_img = heatmap * 0.4 + img * 0.6
        return np.uint8(overlayed_img)
