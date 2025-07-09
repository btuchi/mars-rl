"""Centralized feature extraction using ResNet-18 for visual diversity"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, List
from .trajectory import DiffusionTrajectory

class FeatureExtractor:
    """ResNet-18 based feature extraction for visual diversity"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Load ResNet-18 for image features
        print("Loading ResNet-18 model for image features...")
        self.resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer to get features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet = self.resnet.to(device)
        self.resnet.eval()
        
        # ResNet preprocessing (ImageNet normalization)
        self.resnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ ResNet-18 model loaded")
        
        # Feature dimensions
        self.image_dim = 512  # ResNet-18 feature dimension
    
    def extract_image_features(self, image: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        """Extract ResNet-18 features from image"""
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu()
            image = torch.clamp(image, 0, 1)
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        
        # Preprocess for ResNet and extract features
        image_tensor = self.resnet_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.resnet(image_tensor)
            # ResNet output is [batch_size, 512, 1, 1], flatten to [batch_size, 512]
            features = features.flatten(start_dim=1)
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def extract_trajectory_features(self, trajectory: DiffusionTrajectory) -> np.ndarray:
        """Extract features from final image of trajectory"""
        return self.extract_image_features(trajectory.final_image)