"""
Routing Classifier Model using MobileNetV2.

This module implements a lightweight binary classifier that routes images
to either LCDNet (sparse scenes) or CSRNet (dense scenes) based on
predicted scene density.

Architecture:
- Backbone: MobileNetV2 (pretrained on ImageNet, ~3.5M params)
- Head: AdaptiveAvgPool -> FC(1280->256) -> ReLU -> Dropout -> FC(256->2)
- Output: 2-class logits (class 0=LCDNet, class 1=CSRNet)

Design Choices:
- MobileNetV2: Fast inference (~5ms), edge-deployable, good pretrained features
- Binary classification: Hard routing (one model per image, not ensemble)
- Pretrained backbone: Faster convergence, better generalization

Author: Thesis Implementation - Phase 2 Part 3
"""

import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from routing import config_routing
except ImportError:
    import config_routing


class RoutingClassifier(nn.Module):
    """
    Lightweight classifier to route images to LCDNet or CSRNet.
    
    The model predicts whether an image contains a sparse crowd (route to 
    LCDNet) or a dense crowd (route to CSRNet) based on visual features.
    
    Attributes:
        backbone: MobileNetV2 feature extractor (frozen or trainable).
        classifier: Classification head for 2-class prediction.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = None,
        freeze_backbone: bool = False
    ):
        """
        Initialize the routing classifier.
        
        Args:
            num_classes: Number of output classes (default: 2).
            pretrained: Whether to use pretrained MobileNetV2 weights.
            dropout: Dropout rate for classification head.
            freeze_backbone: Whether to freeze backbone weights.
        """
        super(RoutingClassifier, self).__init__()
        
        if dropout is None:
            dropout = config_routing.ROUTER_DROPOUT
        
        # Load MobileNetV2 backbone with pretrained weights
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Get the number of features from the last conv layer
        # MobileNetV2 outputs 1280 features after global pooling
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        # Optionally freeze backbone for fine-tuning only the head
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters except the classifier."""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the routing classifier.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
               Should be normalized with ImageNet mean/std.
               Expected size: 224x224.
        
        Returns:
            Logits tensor of shape (B, 2) for 2-class classification.
            Class 0 = LCDNet (sparse), Class 1 = CSRNet (dense).
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (0 or 1).
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        
        Returns:
            Predicted class indices of shape (B,).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        
        Returns:
            Probability tensor of shape (B, 2).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_router(
    device: str = None,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> RoutingClassifier:
    """
    Factory function to create and initialize the routing classifier.
    
    Args:
        device: Device to place model on. Uses config default if None.
        pretrained: Whether to use pretrained MobileNetV2 weights.
        freeze_backbone: Whether to freeze backbone weights.
    
    Returns:
        Initialized RoutingClassifier model on specified device.
    """
    if device is None:
        device = config_routing.DEVICE
    
    model = RoutingClassifier(
        num_classes=config_routing.NUM_CLASSES,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    print(f"Router created with {model.count_parameters():,} trainable parameters")
    print(f"Device: {device}")
    
    return model


def load_router(checkpoint_path: str = None, device: str = None) -> RoutingClassifier:
    """
    Load a trained routing classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file. Uses config default if None.
        device: Device to place model on. Uses config default if None.
    
    Returns:
        Loaded RoutingClassifier model in eval mode.
    """
    if checkpoint_path is None:
        checkpoint_path = config_routing.ROUTER_BEST_PATH
    if device is None:
        device = config_routing.DEVICE
    
    model = RoutingClassifier(
        num_classes=config_routing.NUM_CLASSES,
        pretrained=False  # Don't need pretrained weights, loading from checkpoint
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Router loaded from: {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test the router model
    print("Testing RoutingClassifier...")
    print("=" * 50)
    
    device = config_routing.DEVICE
    model = create_router(device=device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        predictions = model.predict(dummy_input)
        probabilities = model.predict_proba(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output logits shape: {output.shape}")
    print(f"Predictions: {predictions.cpu().numpy()}")
    print(f"Probabilities:\n{probabilities.cpu().numpy()}")
    print("=" * 50)
    print("Router test passed!")
