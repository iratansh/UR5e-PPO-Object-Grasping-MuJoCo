"""
CNN Feature Extractor for Sim-to-Real Transfer
Properly handles RGBD input from wrist-mounted camera
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SimToRealCNNExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor optimized for sim-to-real transfer
    Handles robot state + RGBD camera data
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, camera_resolution: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Calculate observation splits
        self.camera_resolution = camera_resolution  # Use provided resolution
        self.camera_channels = 4  # RGBD
        self.camera_pixels = self.camera_resolution ** 2 * self.camera_channels
        
        total_obs = observation_space.shape[0]
        self.kinematic_dims = total_obs - self.camera_pixels
        
        print(f" CNN Feature Extractor Configuration:")
        print(f"   Total observation: {total_obs}")
        print(f"   Kinematic features: {self.kinematic_dims}")
        print(f"   Camera pixels: {self.camera_pixels}")
        print(f"   Camera resolution: {self.camera_resolution}x{self.camera_resolution}")
        
        # RGB processing backbone
        self.rgb_backbone = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Feature blocks
            self._make_conv_block(32, 64, stride=2),
            self._make_conv_block(64, 128, stride=2),
            self._make_conv_block(128, 256, stride=2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        # Depth processing backbone (specialized for depth)
        self.depth_backbone = nn.Sequential(
            # Depth-specific initial layer
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Depth features
            self._make_conv_block(16, 32, stride=2),
            self._make_conv_block(32, 64, stride=2),
            self._make_conv_block(64, 128, stride=2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        # Calculate feature dimensions
        with torch.no_grad():
            sample_rgb = torch.zeros(1, 3, self.camera_resolution, self.camera_resolution)
            sample_depth = torch.zeros(1, 1, self.camera_resolution, self.camera_resolution)
            rgb_features = self.rgb_backbone(sample_rgb).shape[1]
            depth_features = self.depth_backbone(sample_depth).shape[1]
        
        # Kinematic feature processing
        self.kinematic_processor = nn.Sequential(
            nn.Linear(self.kinematic_dims, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        # Feature fusion
        fusion_input_dim = rgb_features + depth_features + 128
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )
        
        print(f"   RGB features: {rgb_features}")
        print(f"   Depth features: {depth_features}")
        print(f"   Fusion input: {fusion_input_dim}")
        print(f"   Output features: {features_dim}")
        
    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a convolutional block with residual connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split observations
        kinematic_data = observations[:, :self.kinematic_dims]
        camera_data = observations[:, self.kinematic_dims:]
        
        # Reshape camera data
        camera_data = camera_data.view(
            batch_size, self.camera_resolution, self.camera_resolution, self.camera_channels
        ).permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Split RGB and depth
        rgb_data = camera_data[:, :3, :, :]
        depth_data = camera_data[:, 3:4, :, :]
        
        # Process each modality
        rgb_features = self.rgb_backbone(rgb_data)
        depth_features = self.depth_backbone(depth_data)
        kinematic_features = self.kinematic_processor(kinematic_data)
        
        # Fuse features
        combined_features = torch.cat([rgb_features, depth_features, kinematic_features], dim=1)
        fused_features = self.fusion_network(combined_features)
        
        return fused_features