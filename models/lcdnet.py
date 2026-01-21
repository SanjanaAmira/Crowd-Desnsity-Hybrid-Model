"""
LCDNet: Lightweight Crowd Density Network.

This module implements LCDNet, a lightweight architecture for crowd density
estimation using depthwise separable convolutions. The network follows an
encoder-decoder structure with skip connections to preserve spatial details.

Key Features:
- Depthwise separable convolutions: Significantly reduce parameters
  compared to standard convolutions while maintaining performance.
- Encoder-decoder structure: Progressively downsamples to capture context,
  then upsamples to produce full-resolution density maps.
- Skip connections: Preserve spatial details by connecting encoder and
  decoder at matching resolutions.

Architecture Overview:
    Input Image (3, H, W)
         ↓
    [Encoder Stage 1] → Skip 1
         ↓
    [Encoder Stage 2] → Skip 2
         ↓
    [Encoder Stage 3] → Skip 3
         ↓
    [Encoder Stage 4] (Bottleneck)
         ↓
    [Decoder Stage 4] + Skip 3
         ↓
    [Decoder Stage 3] + Skip 2
         ↓
    [Decoder Stage 2] + Skip 1
         ↓
    [Decoder Stage 1]
         ↓
    Output Density Map (1, H, W)

Author: Thesis Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    
    This is a factorized convolution that splits a standard convolution into:
    1. Depthwise convolution: Applies a single filter per input channel
    2. Pointwise convolution: 1x1 convolution to combine channel information
    
    Benefits:
    - Reduces parameters: Standard 3x3 conv has k²×Cin×Cout parameters
      Depthwise separable has k²×Cin + Cin×Cout parameters
    - Reduces computation while maintaining representational power
    
    For example, a 3x3 conv from 64 to 128 channels:
    - Standard: 3×3×64×128 = 73,728 parameters
    - Depthwise separable: 3×3×64 + 64×128 = 576 + 8,192 = 8,768 parameters
    - That's ~8.4x fewer parameters!
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        """
        Initialize depthwise separable convolution.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the depthwise kernel.
            stride: Stride for the depthwise convolution.
            padding: Padding for the depthwise convolution.
            bias: Whether to use bias in convolutions.
        """
        super().__init__()
        
        # Depthwise convolution: each input channel is convolved separately
        # groups=in_channels means each channel gets its own filter
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # This makes it depthwise
            bias=bias
        )
        
        # Pointwise convolution: 1x1 conv to mix channel information
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        
        # Batch normalization for stable training
        self.bn = nn.BatchNorm2d(out_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depthwise separable conv."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block with two depthwise separable convolutions and optional pooling.
    
    Each encoder block:
    1. Applies two DSConv layers to extract features
    2. Optionally downsamples via max pooling
    
    The output before pooling is saved for skip connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True
    ):
        """
        Initialize encoder block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            pool: Whether to apply max pooling for downsampling.
        """
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2) if pool else None
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            Tuple of (pooled_output, skip_connection).
            If pool=False, returns (output, output).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x  # Save for skip connection
        
        if self.pool is not None:
            x = self.pool(x)
        
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling, skip connection, and two DSConv layers.
    
    Each decoder block:
    1. Upsamples the input by 2x using bilinear interpolation
    2. Concatenates with skip connection from encoder
    3. Applies two DSConv layers to refine features
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of channels from previous decoder stage.
            skip_channels: Number of channels from skip connection.
            out_channels: Number of output channels.
        """
        super().__init__()
        
        # After concatenation, we have in_channels + skip_channels
        self.conv1 = DepthwiseSeparableConv(
            in_channels + skip_channels,
            out_channels
        )
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input tensor from previous decoder stage.
            skip: Skip connection tensor from corresponding encoder stage.
        
        Returns:
            Decoded features tensor.
        """
        # Upsample to match skip connection size
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class LCDNet(nn.Module):
    """
    Lightweight Crowd Density Network for crowd counting.
    
    LCDNet is an encoder-decoder network that predicts pixel-wise density maps
    from input crowd images. Key design choices:
    
    1. Depthwise Separable Convolutions: Reduce parameters by ~8x while
       maintaining accuracy, making the model efficient.
    
    2. Encoder-Decoder with Skip Connections: The encoder captures multi-scale
       context, while the decoder recovers spatial details. Skip connections
       prevent loss of fine-grained location information.
    
    3. Single-Channel Output: Produces a density map where summing all pixels
       gives the estimated person count.
    
    Input: RGB image tensor of shape (B, 3, H, W)
    Output: Density map tensor of shape (B, 1, H, W)
    
    The output has the same spatial resolution as the input.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        initial_filters: int = None,
        num_stages: int = None,
        dropout_rate: float = None
    ):
        """
        Initialize LCDNet.
        
        Args:
            in_channels: Number of input channels (3 for RGB images).
            initial_filters: Number of filters in first encoder stage.
                            Doubled at each subsequent stage.
            num_stages: Number of encoder/decoder stages.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        # Use config defaults if not specified
        if initial_filters is None:
            initial_filters = config.INITIAL_FILTERS
        if num_stages is None:
            num_stages = config.NUM_STAGES
        if dropout_rate is None:
            dropout_rate = config.DROPOUT_RATE
        
        self.num_stages = num_stages
        
        # Calculate channel sizes for each stage
        # [32, 64, 128, 256] for initial_filters=32, num_stages=4
        encoder_channels = [initial_filters * (2 ** i) for i in range(num_stages)]
        
        # Build encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(encoder_channels):
            # Last encoder stage doesn't pool (it's the bottleneck)
            pool = (i < num_stages - 1)
            self.encoders.append(EncoderBlock(in_ch, out_ch, pool=pool))
            in_ch = out_ch
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Build decoder (reverse order of encoder channels)
        self.decoders = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]  # [256, 128, 64, 32]
        
        for i in range(num_stages - 1):
            in_ch = decoder_channels[i]
            skip_ch = decoder_channels[i + 1]
            out_ch = decoder_channels[i + 1]
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        # Final convolution to produce single-channel density map
        # Using standard conv here for precise output
        self.final_conv = nn.Sequential(
            nn.Conv2d(initial_filters, initial_filters // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filters // 2, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # Density values must be non-negative
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LCDNet.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W).
        
        Returns:
            Density map tensor of shape (B, 1, H, W).
            The sum of the density map approximates the person count.
        """
        # Store skip connections
        skips = []
        
        # Encoder forward pass
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x)
            if i < self.num_stages - 1:  # Don't store last (bottleneck) skip
                skips.append(skip)
        
        # Apply dropout at bottleneck
        x = self.dropout(x)
        
        # Decoder forward pass with skip connections
        # skips are in order [skip1, skip2, skip3], need to use in reverse
        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])
        
        # Final convolution for density output
        density = self.final_conv(x)
        
        return density
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: str = None) -> LCDNet:
    """
    Factory function to create and initialize LCDNet.
    
    Args:
        device: Device to place model on. Uses config default if None.
    
    Returns:
        Initialized LCDNet model on specified device.
    """
    if device is None:
        device = config.DEVICE
    
    model = LCDNet(
        in_channels=3,
        initial_filters=config.INITIAL_FILTERS,
        num_stages=config.NUM_STAGES,
        dropout_rate=config.DROPOUT_RATE
    )
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing LCDNet architecture...")
    
    # Create model
    model = create_model("cpu")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    num_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output min: {output.min().item():.6f}")
    print(f"Output max: {output.max().item():.6f}")
    
    # Verify output is non-negative (density values)
    assert (output >= 0).all(), "Density values should be non-negative!"
    
    # Verify output spatial dimensions match input
    assert output.shape[2:] == dummy_input.shape[2:], \
        "Output spatial dimensions should match input!"
    
    print("\nLCDNet test passed!")
