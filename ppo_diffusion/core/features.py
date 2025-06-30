"""Centralized feature extraction using a single CLIP instance"""

import torch
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, List
from .trajectory import DiffusionTrajectory

class FeatureExtractor:
    """Centralized CLIP feature extraction to avoid duplicate model loading"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Loading CLIP model for feature extraction...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        print("✅ CLIP model loaded")
    
    def extract_text_features(self, prompt: str) -> np.ndarray:
        """Extract CLIP features from text prompt"""
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def extract_image_features(self, image: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        """Extract CLIP features from image"""
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu()
            image = torch.clamp(image, 0, 1)
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        
        # Preprocess and extract features
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def extract_trajectory_features(self, trajectory: DiffusionTrajectory) -> np.ndarray:
        """Extract features from final image of trajectory"""
        return self.extract_image_features(trajectory.final_image)
    
    def calculate_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """Calculate similarity between image and text features"""
        with torch.no_grad():

            # Convert to tensors if needed
            if isinstance(image_features, np.ndarray):
                image_features = torch.from_numpy(image_features).to(self.device)
            if isinstance(text_features, np.ndarray):
                text_features = torch.from_numpy(text_features).to(self.device)
            
            # Ensure proper dimensions
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            if text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = torch.cosine_similarity(image_features, text_features).item()
            return similarity