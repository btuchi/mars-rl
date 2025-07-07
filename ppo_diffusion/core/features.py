"""Centralized feature extraction using CLIP for text and DINO for images"""

import torch
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, List
from .trajectory import DiffusionTrajectory

class FeatureExtractor:
    """Hybrid feature extraction: CLIP for text, DINO for images"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Load CLIP for text features
        print("Loading CLIP model for text features...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        print("✅ CLIP model loaded")
        
        # Load DINO for image features
        print("Loading DINO model for image features...")
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()
        
        # DINO preprocessing
        self.dino_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ DINO model loaded")
        
        # Feature dimensions
        self.text_dim = 512  # CLIP text features
        self.image_dim = 768  # DINO image features
        
        # Create projection layer to align dimensions for similarity calculation
        self.image_projector = torch.nn.Linear(self.image_dim, self.text_dim).to(device)
        print("✅ Feature alignment projector created")
    
    def extract_text_features(self, prompt: str) -> np.ndarray:
        """Extract CLIP features from text prompt"""
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def extract_image_features(self, image: Union[torch.Tensor, Image.Image]) -> np.ndarray:
        """Extract DINO features from image"""
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).cpu()
            image = torch.clamp(image, 0, 1)
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        
        # Preprocess for DINO and extract features
        image_tensor = self.dino_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.dino_model(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def extract_trajectory_features(self, trajectory: DiffusionTrajectory) -> np.ndarray:
        """Extract features from final image of trajectory"""
        return self.extract_image_features(trajectory.final_image)
    
    def calculate_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """Calculate similarity between DINO image and CLIP text features"""
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
            
            # Project DINO image features (768D) to CLIP text space (512D)
            if image_features.shape[-1] == self.image_dim:  # DINO features
                image_features_projected = self.image_projector(image_features)
            else:
                image_features_projected = image_features  # Already projected or wrong dim
            
            # Normalize features
            image_features_projected = image_features_projected / image_features_projected.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = torch.cosine_similarity(image_features_projected, text_features).item()
            return similarity