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

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def save_activation(module, input, output):
            self.activations = output.detach()

        self.handlers.append(self.target_layer.register_forward_hook(save_activation))
        self.handlers.append(self.target_layer.register_full_backward_hook(save_gradient))

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate localized heatmap for a specific class.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # Weight the channels by the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def visualize_on_image(self, img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        Overlay heatmap on the original image.
        """
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert BGR heatmap to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlayed_img = heatmap * 0.4 + img * 0.6
        return np.uint8(overlayed_img)
