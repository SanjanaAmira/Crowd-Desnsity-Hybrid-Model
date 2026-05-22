"""
MobileCount: A Lightweight Encoder-Decoder Network for Crowd Density Estimation.

This model uses a pre-trained MobileNetV2 backbone (features 0-13) as the encoder,
a dilated convolution block for context extraction, and a skip-connection from
feature map layer 6 (resolution 1/8) to produce high-quality density maps
at 1/8 resolution (e.g. 48x48 output for 384x384 input) with under 2M parameters.

Author: Thesis Implementation - Phase 3 Stage 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileCount(nn.Module):
    """
    Lightweight crowd density estimator based on MobileNetV2 features.
    
    Attributes:
        features1 (nn.Sequential): First stage of encoder (1/8 resolution).
        features2 (nn.Sequential): Second stage of encoder (1/16 resolution).
        dilated_blocks (nn.Sequential): Dilated convolutions to capture multi-scale context.
        fuse (nn.Sequential): Fuses skip connection from 1/8 features with decoder features.
        output_head (nn.Sequential): Final 1x1 conv to produce single-channel density map.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load backbone MobileNetV2
        if pretrained:
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            backbone = models.mobilenet_v2(weights=None)
            
        # Encoder Stage 1: layers 0 to 6 (outputs channel=32, stride=8)
        self.features1 = nn.Sequential(*list(backbone.features.children())[:7])
        
        # Encoder Stage 2: layers 7 to 13 (outputs channel=96, stride=16)
        self.features2 = nn.Sequential(*list(backbone.features.children())[7:14])
        
        # Dilated convolutions on 1/16 resolution features (dilation=2)
        # Dilated convolution extracts multi-scale features without losing spatial resolution
        self.dilated_blocks = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion block
        # Combines Stage 1 skip connection (32 channels) + upsampled Stage 2 (96 channels)
        # Total concatenated channels = 32 + 96 = 128
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output Head: project to 1 channel (density map)
        self.output_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # Density must be non-negative
        )
        
        # Initialize decoder weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize decoder conv weights using standard normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m not in self.features1 and m not in self.features2:
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m not in self.features1 and m not in self.features2:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor: Density map tensor of shape (B, 1, H/8, W/8).
        """
        # Encoder Stage 1 (1/8 resolution skip connection)
        skip = self.features1(x)  # Shape: (B, 32, H/8, W/8)
        
        # Encoder Stage 2 (1/16 resolution)
        feat = self.features2(skip)  # Shape: (B, 96, H/16, W/16)
        
        # Apply context extraction (dilated convolutions)
        feat = self.dilated_blocks(feat)  # Shape: (B, 96, H/16, W/16)
        
        # Bilinear upsample to double resolution (1/16 -> 1/8)
        upsampled = F.interpolate(
            feat,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False
        )  # Shape: (B, 96, H/8, W/8)
        
        # Concatenate skip connection and upsampled features
        fused = torch.cat([skip, upsampled], dim=1)  # Shape: (B, 128, H/8, W/8)
        
        # Apply fusion layers
        fused = self.fuse(fused)  # Shape: (B, 32, H/8, W/8)
        
        # Generate density map
        density = self.output_head(fused)  # Shape: (B, 1, H/8, W/8)
        
        return density
        
    def count_parameters(self) -> int:
        """Count the total number of parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing MobileCount Architecture...")
    model = MobileCount(pretrained=False)
    
    num_params = model.count_parameters()
    print(f"Total parameters: {num_params:,}")
    
    # Verify parameter count is under 2M
    assert num_params < 2000000, f"Parameter count {num_params} exceeds 2M limit!"
    print("Parameter limit check passed (< 2M).")
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
        
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 1, 48, 48), f"Incorrect output shape: {out.shape}"
    print("Output shape check passed.")
    print("MobileCount architecture verification successful!")
