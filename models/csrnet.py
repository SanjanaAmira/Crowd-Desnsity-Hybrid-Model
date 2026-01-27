"""
CSRNet: Congested Scene Recognition Network for Crowd Counting.

This module implements CSRNet, a deep neural network for accurate crowd counting
in highly congested scenes. The network uses:
1. VGG-16 frontend (first 10 conv layers) for feature extraction
2. Dilated convolution backend for density map generation

Key Features:
- Uses pretrained VGG-16 weights for robust feature extraction
- Dilated convolutions in backend preserve spatial resolution
- Outputs 1/8 resolution density map (sum = estimated count)

Reference:
    Li et al., "CSRNet: Dilated Convolutional Neural Networks for 
    Understanding the Highly Congested Scenes", CVPR 2018

Architecture Overview:
    Input Image (3, H, W)
         ↓
    [VGG-16 Frontend: conv1_1 → conv4_3]
    - 10 convolutional layers
    - 3 max pooling layers
    - Output: (512, H/8, W/8)
         ↓
    [Dilated Backend]
    - 6 dilated conv layers (dilation=2)
    - Gradually reduce channels: 512 → 256 → 128 → 64
         ↓
    [1x1 Conv Output Layer]
         ↓
    Output Density Map (1, H/8, W/8)

Author: Thesis Implementation - Phase 2 Part 2
"""

import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CSRNet(nn.Module):
    """
    CSRNet for dense crowd counting.
    
    CSRNet uses a VGG-16 frontend pretrained on ImageNet followed by
    a backend with dilated convolutions. The dilated convolutions
    maintain a large receptive field while preserving spatial resolution,
    which is crucial for accurate density estimation.
    
    The network outputs a density map at 1/8 of the input resolution.
    Summing all pixels in the density map gives the estimated person count.
    
    Attributes:
        frontend: VGG-16 first 10 conv layers for feature extraction.
        backend: Dilated convolution layers for density estimation.
        output_layer: 1x1 conv to produce single-channel density map.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize CSRNet.
        
        Args:
            pretrained: If True, use pretrained VGG-16 weights for frontend.
                       Highly recommended for better convergence.
        """
        super().__init__()
        
        # =====================================================================
        # FRONTEND: VGG-16 Feature Extractor (conv1_1 to conv4_3)
        # =====================================================================
        # We use the first 10 convolutional layers of VGG-16:
        # - conv1: 2 layers (64 filters)
        # - conv2: 2 layers (128 filters)  
        # - conv3: 3 layers (256 filters)
        # - conv4: 3 layers (512 filters) <- we stop here
        # 
        # After conv4, spatial resolution is 1/8 of input (3 max pools)
        # This provides a good balance between context and localization
        
        # Load pretrained VGG-16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Extract frontend layers: features[0:23] gives conv1 through conv4_3
        # VGG16 features structure:
        # 0-3: conv1 (64) + pool
        # 4-8: conv2 (128) + pool  
        # 9-15: conv3 (256) + pool
        # 16-22: conv4 (512) - NO pool at end
        frontend_layers = list(vgg.features.children())[:23]
        
        self.frontend = nn.Sequential(*frontend_layers)
        
        # Freeze frontend batch norm layers if any (VGG-16 doesn't have BN)
        # but we keep this for potential future modifications
        
        # =====================================================================
        # BACKEND: Dilated Convolution Layers
        # =====================================================================
        # The backend uses dilated convolutions (also called atrous convolutions)
        # with dilation rate of 2. This:
        # - Increases receptive field without losing resolution
        # - Captures multi-scale context for dense crowd scenes
        # - Maintains spatial information for accurate localization
        #
        # Structure: 512 → 512 → 512 → 256 → 128 → 64
        
        self.backend = nn.Sequential(
            # Dilated conv block 1: maintain 512 channels
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Dilated conv block 2: maintain 512 channels
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Dilated conv block 3: maintain 512 channels
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Dilated conv block 4: reduce to 256 channels
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Dilated conv block 5: reduce to 128 channels
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Dilated conv block 6: reduce to 64 channels
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        
        # =====================================================================
        # OUTPUT LAYER: 1x1 Convolution for Density Map
        # =====================================================================
        # Final 1x1 conv produces single-channel density map
        # No activation - density values can be any positive number
        # (we'll apply ReLU to ensure non-negative output)
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Initialize backend and output layer weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights for backend and output layers.
        
        Uses Gaussian initialization as in the original CSRNet paper.
        Frontend weights are already pretrained from VGG-16.
        """
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize output layer
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CSRNet.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
               Should be normalized with ImageNet mean/std.
        
        Returns:
            Density map tensor of shape (B, 1, H/8, W/8).
            The sum of all pixels approximates the person count.
            
        Note:
            Output resolution is 1/8 of input due to 3 max pooling layers.
            For input (B, 3, 384, 384), output is (B, 1, 48, 48).
        """
        # Frontend: extract features with VGG-16
        x = self.frontend(x)
        
        # Backend: dilated convolutions for density estimation
        x = self.backend(x)
        
        # Output layer: produce density map
        x = self.output_layer(x)
        
        # Apply ReLU to ensure non-negative density values
        # (density represents people per pixel, must be >= 0)
        x = torch.relu(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_frontend(self):
        """
        Freeze frontend (VGG-16) parameters.
        
        Useful for fine-tuning when you want to train only the backend.
        This can help prevent overfitting on small datasets.
        """
        for param in self.frontend.parameters():
            param.requires_grad = False
    
    def unfreeze_frontend(self):
        """Unfreeze frontend parameters for full training."""
        for param in self.frontend.parameters():
            param.requires_grad = True


def create_csrnet(device: str = None, pretrained: bool = True) -> CSRNet:
    """
    Factory function to create and initialize CSRNet.
    
    Args:
        device: Device to place model on. Defaults to CUDA if available.
        pretrained: Whether to use pretrained VGG-16 weights.
    
    Returns:
        Initialized CSRNet model on specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CSRNet(pretrained=pretrained)
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing CSRNet Architecture")
    print("=" * 60)
    
    # Create model
    model = create_csrnet("cpu", pretrained=False)  # False for faster testing
    
    # Print model summary
    print("\nModel Architecture:")
    print("-" * 40)
    print(f"Frontend layers: {len(list(model.frontend.children()))}")
    print(f"Backend layers: {len(list(model.backend.children()))}")
    
    # Count parameters
    total_params = model.count_parameters()
    frontend_params = sum(p.numel() for p in model.frontend.parameters())
    backend_params = sum(p.numel() for p in model.backend.parameters())
    output_params = sum(p.numel() for p in model.output_layer.parameters())
    
    print(f"\nParameter Count:")
    print(f"  Frontend (VGG-16): {frontend_params:,}")
    print(f"  Backend (Dilated): {backend_params:,}")
    print(f"  Output Layer:      {output_params:,}")
    print(f"  Total:             {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 384, 384
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    print(f"\nForward Pass Test:")
    print(f"  Input shape:  {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output min:   {output.min().item():.6f}")
    print(f"  Output max:   {output.max().item():.6f}")
    print(f"  Output mean:  {output.mean().item():.6f}")
    
    # Verify output properties
    assert output.shape == (batch_size, 1, height // 8, width // 8), \
        f"Expected shape {(batch_size, 1, height // 8, width // 8)}, got {output.shape}"
    assert (output >= 0).all(), "Density values should be non-negative!"
    
    print("\n" + "=" * 60)
    print("CSRNet test passed!")
    print("=" * 60)
