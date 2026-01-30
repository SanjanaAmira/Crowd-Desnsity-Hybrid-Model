"""
Hybrid Inference System for Crowd Density Estimation.

This module implements the core hybrid inference pipeline that dynamically
routes images to either LCDNet or CSRNet based on a routing classifier.

Inference Flow:
    Input Image
         ↓
    Routing Classifier (MobileNetV2, 224×224)
         ↓
    Route Decision (sparse/dense)
       ╱     ╲
   LCDNet    CSRNet (frozen, 384×384)
       ↓         ↓
    Density Map
         ↓
    sum() → Count

Key Features:
- Only ONE model runs per image (hard routing, not ensemble)
- Both density models are frozen (no gradient computation)
- Fast inference with minimal overhead from router

Author: Thesis Implementation - Phase 2 Part 3
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import config_routing
from routing.router import load_router

# Import density estimation models
from models.lcdnet import LCDNet
from models.csrnet import CSRNet
import config


class HybridDensityEstimator:
    """
    Hybrid crowd density estimator using routing-based model selection.
    
    This class orchestrates:
    1. A lightweight routing classifier to decide if scene is sparse/dense
    2. LCDNet for sparse scenes (efficient inference)
    3. CSRNet for dense scenes (accurate estimation)
    
    Only one density model runs per image (hard routing).
    
    Attributes:
        router: MobileNetV2-based routing classifier.
        lcdnet: LCDNet model for sparse scenes.
        csrnet: CSRNet model for dense scenes.
        device: Computation device (cuda/cpu).
    """
    
    def __init__(
        self,
        router_path: str = None,
        lcdnet_path: str = None,
        csrnet_path: str = None,
        device: str = None
    ):
        """
        Initialize the hybrid estimator.
        
        Args:
            router_path: Path to router checkpoint. Uses config default if None.
            lcdnet_path: Path to LCDNet checkpoint. Uses config default if None.
            csrnet_path: Path to CSRNet checkpoint. Uses config default if None.
            device: Computation device. Uses config default if None.
        """
        if router_path is None:
            router_path = config_routing.ROUTER_BEST_PATH
        if lcdnet_path is None:
            lcdnet_path = config_routing.LCDNET_CHECKPOINT
        if csrnet_path is None:
            csrnet_path = config_routing.CSRNET_CHECKPOINT
        if device is None:
            device = config_routing.DEVICE
        
        self.device = device
        
        print("Loading Hybrid Density Estimator...")
        print("=" * 50)
        
        # Load routing classifier
        print("[1/3] Loading Router...")
        self.router = self._load_router(router_path)
        
        # Load LCDNet (sparse scenes)
        print("[2/3] Loading LCDNet...")
        self.lcdnet = self._load_lcdnet(lcdnet_path)
        
        # Load CSRNet (dense scenes)
        print("[3/3] Loading CSRNet...")
        self.csrnet = self._load_csrnet(csrnet_path)
        
        # Image transforms
        self._build_transforms()
        
        print("=" * 50)
        print("Hybrid estimator ready!")
    
    def _load_router(self, path: str):
        """Load and freeze the routing classifier."""
        router = load_router(checkpoint_path=path, device=self.device)
        router.eval()
        for param in router.parameters():
            param.requires_grad = False
        print(f"  Router loaded from: {path}")
        return router
    
    def _load_lcdnet(self, path: str):
        """Load and freeze LCDNet."""
        model = LCDNet()
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"  LCDNet loaded from: {path}")
        return model
    
    def _load_csrnet(self, path: str):
        """Load and freeze CSRNet."""
        model = CSRNet(pretrained=False)
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"  CSRNet loaded from: {path}")
        return model
    
    def _build_transforms(self):
        """Build image transformation pipelines."""
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Router transform (224x224)
        self.router_transform = transforms.Compose([
            transforms.Resize(config_routing.ROUTER_INPUT_SIZE),
            transforms.ToTensor(),
            normalize
        ])
        
        # Density model transform (384x384)
        self.density_transform = transforms.Compose([
            transforms.Resize(config_routing.DENSITY_INPUT_SIZE),
            transforms.ToTensor(),
            normalize
        ])
    
    def predict(
        self,
        image: Image.Image,
        return_density_map: bool = False
    ) -> Dict[str, Any]:
        """
        Predict crowd count for a single image using hybrid routing.
        
        Args:
            image: PIL Image (RGB).
            return_density_map: Whether to return the density map.
        
        Returns:
            Dictionary containing:
            - 'count': Predicted person count.
            - 'model': Model used ('LCDNet' or 'CSRNet').
            - 'routing_prob': Router confidence for the chosen model.
            - 'router_time': Time spent on routing (seconds).
            - 'density_time': Time spent on density estimation (seconds).
            - 'total_time': Total inference time (seconds).
            - 'density_map': (optional) Predicted density map as numpy array.
        """
        start_time = time.time()
        
        # Prepare image for router
        router_input = self.router_transform(image).unsqueeze(0).to(self.device)
        
        # Get routing decision
        router_start = time.time()
        with torch.no_grad():
            router_logits = self.router(router_input)
            router_probs = F.softmax(router_logits, dim=1)
            route_decision = torch.argmax(router_probs, dim=1).item()
            routing_confidence = router_probs[0, route_decision].item()
        router_time = time.time() - router_start
        
        # Prepare image for density model
        density_input = self.density_transform(image).unsqueeze(0).to(self.device)
        
        # Run selected density model
        density_start = time.time()
        with torch.no_grad():
            if route_decision == 0:
                # Sparse scene -> LCDNet
                density_map = self.lcdnet(density_input)
                model_name = "LCDNet"
            else:
                # Dense scene -> CSRNet
                density_map = self.csrnet(density_input)
                model_name = "CSRNet"
        density_time = time.time() - density_start
        
        # Compute count from density map
        count = density_map.sum().item()
        
        # Total time
        total_time = time.time() - start_time
        
        # Build result
        result = {
            'count': count,
            'model': model_name,
            'routing_prob': routing_confidence,
            'router_time': router_time,
            'density_time': density_time,
            'total_time': total_time
        }
        
        if return_density_map:
            result['density_map'] = density_map.squeeze().cpu().numpy()
        
        return result
    
    def predict_batch(
        self,
        images: list,
        return_density_maps: bool = False
    ) -> list:
        """
        Predict crowd counts for a batch of images.
        
        Note: Each image is processed individually due to routing decisions.
        
        Args:
            images: List of PIL Images (RGB).
            return_density_maps: Whether to return density maps.
        
        Returns:
            List of result dictionaries.
        """
        results = []
        for image in images:
            result = self.predict(image, return_density_map=return_density_maps)
            results.append(result)
        return results


def predict_single_image(
    image_path: str,
    router_path: str = None,
    lcdnet_path: str = None,
    csrnet_path: str = None,
    device: str = None
) -> Dict[str, Any]:
    """
    Convenience function to predict count for a single image file.
    
    Args:
        image_path: Path to image file.
        router_path: Path to router checkpoint.
        lcdnet_path: Path to LCDNet checkpoint.
        csrnet_path: Path to CSRNet checkpoint.
        device: Computation device.
    
    Returns:
        Prediction result dictionary.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create estimator
    estimator = HybridDensityEstimator(
        router_path=router_path,
        lcdnet_path=lcdnet_path,
        csrnet_path=csrnet_path,
        device=device
    )
    
    # Predict
    result = estimator.predict(image, return_density_map=True)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid inference test")
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to test image'
    )
    args = parser.parse_args()
    
    print("Testing Hybrid Density Estimator...")
    print("=" * 60)
    
    # Create estimator
    try:
        estimator = HybridDensityEstimator()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have trained the router first:")
        print("  python routing/train_router.py")
        sys.exit(1)
    
    # Test with provided image or find a sample
    if args.image and os.path.exists(args.image):
        test_image_path = args.image
    else:
        # Find a sample image from NWPU-Crowd
        test_image_path = None
        for img_dir in config_routing.NWPU_IMAGE_DIRS:
            if os.path.exists(img_dir):
                for f in os.listdir(img_dir)[:1]:
                    test_image_path = os.path.join(img_dir, f)
                    break
            if test_image_path:
                break
        
        if test_image_path is None:
            print("No test image found!")
            sys.exit(1)
    
    print(f"\nTest image: {test_image_path}")
    
    # Load and predict
    image = Image.open(test_image_path).convert('RGB')
    result = estimator.predict(image, return_density_map=True)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Model used: {result['model']}")
    print(f"Routing confidence: {result['routing_prob']:.2%}")
    print(f"Predicted count: {result['count']:.1f}")
    print(f"Router time: {result['router_time']*1000:.1f} ms")
    print(f"Density time: {result['density_time']*1000:.1f} ms")
    print(f"Total time: {result['total_time']*1000:.1f} ms")
    print("=" * 60)
