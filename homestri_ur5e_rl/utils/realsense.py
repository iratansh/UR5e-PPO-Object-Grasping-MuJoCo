"""
RealSense D435i Camera Integration for MuJoCo
Accurate FOV simulation and proper camera handling
"""

import numpy as np
import mujoco
from typing import Tuple, Optional
import warnings
import os

# Set backend before importing anything else
os.environ["MUJOCO_GL"] = "glfw"

class RealSenseD435iSimulator:
    """
    RealSense D435i simulator with accurate FOV settings
    RGB: 69° × 42° (H×V)
    Depth: 58° × 45° (H×V)
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 camera_name: str = "realsense_rgb",
                 render_resolution: int = 128):
        self.model = model
        self.data = data
        self.resolution = render_resolution
        
        # Find cameras in model
        try:
            self.rgb_camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            self.depth_camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "realsense_depth"
            )
            
            if self.rgb_camera_id < 0:
                raise ValueError(f"RGB camera '{camera_name}' not found")
            
            print(f" Found RGB camera with ID {self.rgb_camera_id}")
            if self.depth_camera_id >= 0:
                print(f" Found depth camera with ID {self.depth_camera_id}")
            else:
                print(f" Depth camera not found, using RGB camera for depth")
                self.depth_camera_id = self.rgb_camera_id
                
        except Exception as e:
            print(f" Camera initialization failed: {e}")
            self.rgb_camera_id = -1
            self.depth_camera_id = -1
            self.scene = None
            self.context = None
            return
        
        # RealSense D435i official parameters from Intel specs
        self.rgb_fov_horizontal = 69.0   # degrees
        self.rgb_fov_vertical = 42.0     # degrees
        self.depth_fov_horizontal = 87.0  # degrees 
        self.depth_fov_vertical = 58.0    # degrees
        
        self.min_depth = 0.28  # meters
        self.max_depth = 3.0   # meters
        self.depth_noise_percent = 0.02  # 2% at 2m
        
        # Initialize rendering components
        self._init_rendering()
        
    def _init_rendering(self):
        """Initialize rendering components with error handling and a warm-up render."""
        try:
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            
            self.viewport = mujoco.MjrRect(0, 0, self.resolution, self.resolution)
            
            # RGB camera setup
            self.rgb_camera = mujoco.MjvCamera()
            self.rgb_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.rgb_camera.fixedcamid = self.rgb_camera_id
            
            # Depth camera setup
            self.depth_camera = mujoco.MjvCamera()
            self.depth_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.depth_camera.fixedcamid = self.depth_camera_id
            
            self.option = mujoco.MjvOption()
            # Explicitly enable all standard visual elements to combat EGL issues
            for i in range(mujoco.mjtVisFlag.mjNVISFLAG):
                self.option.flags[i] = True
            self.option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False # Usually off
            self.option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            self.option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            
            # Pre-allocate buffers
            self.rgb_buffer = np.zeros(
                (self.resolution, self.resolution, 3), dtype=np.uint8
            )
            self.depth_buffer = np.zeros(
                (self.resolution, self.resolution), dtype=np.float32
            )
            
            print(f" RealSense rendering initialized: {self.resolution}x{self.resolution}")
            print(f"   RGB FOV: {self.rgb_fov_horizontal}° × {self.rgb_fov_vertical}°")
            print(f"   Depth FOV: {self.depth_fov_horizontal}° × {self.depth_fov_vertical}°")

        except Exception as e:
            print(f" Rendering initialization failed: {e}")
            self.scene = None
            self.context = None
    
    def _reinit_rendering(self):
        """Reinitialize rendering components if an issue is detected."""
        try:
            if hasattr(self, 'context') and self.context is not None:
                self.context = None
            
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.viewport = mujoco.MjrRect(0, 0, self.resolution, self.resolution)
            
            print("✅ Rendering context reinitialized successfully.")
        except Exception as e:
            print(f"❌ Failed to reinitialize rendering: {e}")
            
    def render_rgbd(self) -> np.ndarray:
        """
        Render RGB-D data. Includes a proactive check to reinitialize context if the RGB buffer is blank.
        Returns: Flattened RGBD array (resolution^2 * 4)
        """
        if self.rgb_camera_id < 0 or self.scene is None or self.context is None:
            return np.zeros(self.resolution * self.resolution * 4, dtype=np.float32)
    
        try:
            # Set buffer to offscreen and render the scene once
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.rgb_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            
            # Read both RGB and Depth buffers from the single render pass
            mujoco.mjr_readPixels(self.rgb_buffer, self.depth_buffer, self.viewport, self.context)

            # Proactive check for blank RGB buffer
            if np.max(self.rgb_buffer) == 0:
                warnings.warn("Blank RGB buffer detected. Attempting to reinitialize and retry render.")
                self._reinit_rendering()
                
                # Retry the render one time after re-initialization
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
                mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.rgb_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                mujoco.mjr_render(self.viewport, self.scene, self.context)
                mujoco.mjr_readPixels(self.rgb_buffer, self.depth_buffer, self.viewport, self.context)
            
            # Process RGB
            rgb_processed = np.flipud(self.rgb_buffer.copy()).astype(np.float32) / 255.0
            
            # Process Depth (already read from the buffer)
            depth_processed = self._process_depth(np.flipud(self.depth_buffer.copy()))
            
            # Combine RGBD
            rgbd = np.zeros((self.resolution, self.resolution, 4), dtype=np.float32)
            rgbd[:, :, :3] = rgb_processed
            rgbd[:, :, 3] = depth_processed
            
            return rgbd.flatten()
            
        except Exception as e:
            print(f" Camera render error: {e}")
            return np.zeros(self.resolution * self.resolution * 4, dtype=np.float32)
    
    def _process_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """Convert MuJoCo depth buffer to realistic depth values"""
        try:
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent
            
            depth_meters = np.zeros_like(raw_depth)
            valid_mask = (raw_depth > 0) & (raw_depth < 1) & np.isfinite(raw_depth)
            
            if np.any(valid_mask):
                depth_meters[valid_mask] = near * far / (far - raw_depth[valid_mask] * (far - near))
            
            depth_meters = self._add_depth_noise(depth_meters)
            depth_meters = self._apply_range_limits(depth_meters)
            return self._normalize_depth(depth_meters)
            
        except Exception as e:
            print(f" Depth processing error: {e}")
            return np.zeros_like(raw_depth, dtype=np.float32)
    
    def _add_depth_noise(self, depth: np.ndarray) -> np.ndarray:
        """Add realistic depth sensor noise"""
        valid_mask = (depth > 0) & np.isfinite(depth)
        if not np.any(valid_mask):
            return depth
        
        depth_noisy = depth.copy()
        
        noise_std = depth[valid_mask] * self.depth_noise_percent
        noise = np.random.normal(0, noise_std)
        depth_noisy[valid_mask] += noise
        
        if np.random.random() < 0.01:
            dropout_mask = np.random.random(depth.shape) < 0.001
            depth_noisy[dropout_mask] = 0
        
        return np.maximum(depth_noisy, 0)
    
    def _apply_range_limits(self, depth: np.ndarray) -> np.ndarray:
        """Apply RealSense D435i range limitations"""
        depth_limited = depth.copy()
        depth_limited[depth < self.min_depth] = 0
        depth_limited[depth > self.max_depth] = 0
        return depth_limited
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth for neural network input"""
        normalized = np.zeros_like(depth, dtype=np.float32)
        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        
        if np.any(valid_mask):
            depth_range = self.max_depth - self.min_depth
            normalized[valid_mask] = (depth[valid_mask] - self.min_depth) / depth_range
            
        return np.clip(normalized, 0, 1)
    
    def get_rgb_image(self) -> np.ndarray:
        """Get only RGB image for visualization"""
        if self.rgb_camera_id < 0 or self.scene is None or self.context is None:
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        
        try:
            mujoco.mjv_updateScene(
                self.model, self.data, self.option, None, self.rgb_camera, 
                mujoco.mjtCatBit.mjCAT_ALL, self.scene
            )
            
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            mujoco.mjr_readPixels(self.rgb_buffer, None, self.viewport, self.context)
            
            return np.flipud(self.rgb_buffer.copy())
            
        except Exception as e:
            print(f" RGB render error: {e}")
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
    
    def is_available(self) -> bool:
        """Check if camera is available and functional"""
        return (self.rgb_camera_id >= 0 and 
                self.scene is not None and 
                self.context is not None)
    
    def close(self):
        """Clean shutdown"""
        try:
            self.context = None
            self.scene = None
        except Exception:
            pass