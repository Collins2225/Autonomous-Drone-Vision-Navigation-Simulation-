"""
perception/depth_estimator.py
==============================
[RESEARCH EXTENSION] Monocular depth estimation using MiDaS.

MENTOR NOTE — Monocular Depth: The Hard Problem
  Our main pipeline uses AirSim's built-in depth camera. But what if you
  only have a single RGB camera (like most drones)?
  
  Monocular depth estimation uses a neural network trained on large datasets
  to predict relative depth from a SINGLE image. This is impressive but
  comes with a key limitation: it produces RELATIVE depth (scene geometry),
  not ABSOLUTE depth (real meters). You can't directly know "that wall is 3m".
  
  To get metric depth from mono:
    1. Use scale recovery with known object sizes (e.g., ArUco markers)
    2. Fuse with IMU/odometry to scale over time (like ORB-SLAM2 mono)
    3. Use specialized metric mono models (e.g., ZoeDepth, Depth Pro)
  
  This module provides a drop-in replacement for the depth camera —
  same interface, different data source.
  
  MiDaS paper: Ranftl et al., "Towards Robust Monocular Depth Estimation:
  Mixing Datasets for Zero-Shot Cross-Dataset Transfer" (TPAMI 2022)

Requirements:
    pip install torch torchvision timm
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class MonocularDepthEstimator:
    """
    Wraps MiDaS for real-time relative depth estimation from RGB.
    
    Use this as a replacement for AirSim depth camera when:
    - Testing on real hardware with only an RGB camera
    - Comparing against ground-truth AirSim depth (for research evaluation)
    
    Usage:
        estimator = MonocularDepthEstimator(model_type="MiDaS_small")
        depth_relative = estimator.estimate(rgb_frame)
        # depth_relative shape: (H, W), values in [0, 1], higher = farther
    """

    # Available model sizes (tradeoff: accuracy vs speed)
    MODELS = {
        "DPT_Large":    "Highest accuracy, slowest (~100ms on GPU)",
        "DPT_Hybrid":   "Good accuracy, moderate speed (~30ms on GPU)",
        "MiDaS_small":  "Lower accuracy, fastest (~5ms on CPU)",
    }

    def __init__(self, model_type: str = "MiDaS_small", device: str = "auto"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not found. Install with: pip install torch torchvision"
            )

        self.model_type = model_type

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[DepthEstimator] Loading {model_type} on {self.device}...")
        self._load_model()
        print("[DepthEstimator] Model loaded ✓")

    def _load_model(self):
        """
        Load MiDaS from torch.hub.
        
        MENTOR NOTE:
          torch.hub downloads models automatically. First run requires
          internet. Subsequent runs use cached weights (~100MB for DPT_Hybrid).
          Cache location: ~/.cache/torch/hub/
        """
        # Load model
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            self.model_type,
            trust_repo=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Load matching transform (preprocessing pipeline for this model)
        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        )
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """
        Estimate relative depth from an RGB image.
        
        Args:
            rgb: H×W×3 uint8 numpy array (RGB, not BGR!)
                 Note: OpenCV uses BGR. Convert with cv2.cvtColor if needed.
        
        Returns:
            depth_relative: H×W float32 array, values in roughly [0, 1]
                           Higher values = FARTHER away (inverse of disparity)
                           NOT in real metric units!
        
        MENTOR NOTE — Relative vs Absolute Depth:
          MiDaS output is an INVERSE DEPTH / DISPARITY map.
          Large values = far away. Small values = close.
          
          To convert to approximate metric depth for obstacle detection:
          1. Normalize to [0, 1]
          2. Invert: real_depth_approx = 1.0 / (normalized_inv_depth + eps)
          3. Scale with known scene depth if available
          
          For obstacle avoidance, relative depth is often sufficient:
          we just threshold "what's closest" regardless of exact meters.
        """
        # Preprocess
        input_batch = self.transform(rgb).to(self.device)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize to original image dimensions
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        # Convert to numpy
        output = prediction.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]
        out_min, out_max = output.min(), output.max()
        if out_max > out_min:
            output = (output - out_min) / (out_max - out_min)

        return output

    def estimate_as_metric_proxy(
        self, rgb: np.ndarray, scale: float = 10.0
    ) -> np.ndarray:
        """
        Returns a rough metric-proxy depth by inverting MiDaS output.
        
        NOT accurate in absolute terms but usable for obstacle detection thresholds.
        
        Args:
            rgb:   H×W×3 RGB image
            scale: approximate scene scale in meters (tune for your environment)
        
        Returns:
            H×W float32, rough depth in "meters" (not precise)
        """
        relative = self.estimate(rgb)
        # Invert: MiDaS large values = far, so invert to get distance proxy
        # Add epsilon to avoid division by zero
        metric_proxy = scale / (relative + 0.1)
        return metric_proxy


# ──────────────────────────────────────────────
# DROP-IN REPLACEMENT DEMO
# ──────────────────────────────────────────────

def demo_depth_vs_airsim():
    """
    Research demo: compare MiDaS estimated depth with AirSim ground truth.
    
    Use this to evaluate how well monocular depth works in your environment.
    This is standard practice in depth estimation research papers.
    """
    import matplotlib.pyplot as plt
    from simulation.airsim_client import DroneSimClient

    sim = DroneSimClient(mock_mode=False)
    estimator = MonocularDepthEstimator("MiDaS_small")

    images = sim.get_images()
    rgb_bgr = images["rgb"]
    rgb     = rgb_bgr[:, :, ::-1]   # BGR → RGB for MiDaS
    gt_depth = images["depth"]       # ground truth from AirSim

    # Estimate
    estimated = estimator.estimate_as_metric_proxy(rgb, scale=15.0)

    # Compare
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(rgb);               axes[0].set_title("RGB Input")
    axes[1].imshow(gt_depth, cmap='jet');    axes[1].set_title("GT Depth (AirSim)")
    axes[2].imshow(estimated, cmap='jet');   axes[2].set_title("MiDaS Estimated")
    plt.tight_layout()
    plt.savefig("depth_comparison.png", dpi=150)
    print("Saved depth_comparison.png")


if __name__ == "__main__":
    demo_depth_vs_airsim()
