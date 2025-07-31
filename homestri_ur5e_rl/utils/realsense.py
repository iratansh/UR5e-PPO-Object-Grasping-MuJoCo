"""
RealSense D435i Camera Integration for MuJoCo
FIXED: Proper rendering context handling for parallel environments
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
    FIXED: Lazy context initialization for parallel environments
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
            
        # RealSense D435i official parameters from Intel specs
        self.rgb_fov_horizontal = 69.0   # degrees
        self.rgb_fov_vertical = 42.0     # degrees
        self.depth_fov_horizontal = 87.0  # degrees 
        self.depth_fov_vertical = 58.0    # degrees
        
        self.min_depth = 0.28  # meters
        self.max_depth = 3.0   # meters
        self.depth_noise_percent = 0.02  # 2% at 2m
        
        # FIXED: Delay rendering initialization until first use
        self.scene = None
        self.context = None
        self.viewport = None
        self.rgb_camera = None
        self.depth_camera = None
        self.option = None
        self.rgb_buffer = None
        self.depth_buffer = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """FIXED: Lazy initialization of rendering components"""
        if not self._initialized:
            self._init_rendering()
        
    def _init_rendering(self):
        """Initialize rendering components with better error handling"""
        try:
            # Create scene
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            
            # Create context with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
                    if self.context is not None:
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f" Retry {attempt + 1}/{max_retries} for context creation...")
                        continue
                    else:
                        raise e
                        
            if self.context is None:
                raise RuntimeError("Failed to create rendering context after retries")
                
            # Create viewport
            self.viewport = mujoco.MjrRect(0, 0, self.resolution, self.resolution)
            
            # RGB camera setup
            self.rgb_camera = mujoco.MjvCamera()
            self.rgb_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.rgb_camera.fixedcamid = self.rgb_camera_id
            
            # Depth camera setup
            self.depth_camera = mujoco.MjvCamera()
            self.depth_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.depth_camera.fixedcamid = self.depth_camera_id
            
            # Rendering options
            self.option = mujoco.MjvOption()
            # Enable standard visual elements
            for i in range(mujoco.mjtVisFlag.mjNVISFLAG):
                self.option.flags[i] = True
            self.option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
            self.option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            self.option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            
            # Pre-allocate buffers
            self.rgb_buffer = np.zeros(
                (self.resolution, self.resolution, 3), dtype=np.uint8
            )
            self.depth_buffer = np.zeros(
                (self.resolution, self.resolution), dtype=np.float32
            )
            
            self._initialized = True
            print(f" RealSense rendering initialized: {self.resolution}x{self.resolution}")
            print(f"   RGB FOV: {self.rgb_fov_horizontal}° × {self.rgb_fov_vertical}°")
            print(f"   Depth FOV: {self.depth_fov_horizontal}° × {self.depth_fov_vertical}°")

        except Exception as e:
            print(f" Rendering initialization failed: {e}")
            self.scene = None
            self.context = None
            self._initialized = False
            raise e
    
    def _reinit_rendering(self):
        """Reinitialize rendering components if an issue is detected"""
        try:
            # Clean up old context
            if self.context is not None:
                self.context = None
                
            # Reset initialization flag
            self._initialized = False
            
            # Reinitialize
            self._init_rendering()
            
            if self._initialized:
                print("✅ Rendering context reinitialized successfully.")
            else:
                print("❌ Failed to reinitialize rendering")
                
        except Exception as e:
            print(f"❌ Failed to reinitialize rendering: {e}")
            self._initialized = False
            
    def render_rgbd(self) -> np.ndarray:
        """
        Render RGB-D data with lazy initialization and error recovery
        Returns: Flattened RGBD array (resolution^2 * 4)
        """
        # Ensure initialized
        self._ensure_initialized()
        
        # Check if camera is valid
        if self.rgb_camera_id < 0:
            return np.zeros(self.resolution * self.resolution * 4, dtype=np.float32)
        
        # Check if context is valid
        if self.context is None:
            print(" Context is None, attempting reinitialization...")
            self._reinit_rendering()
            if self.context is None:
                # Return dummy data if still failed
                return np.zeros(self.resolution * self.resolution * 4, dtype=np.float32)
    
        try:
            # Set buffer to offscreen and render the scene
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.rgb_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            
            # Read both RGB and Depth buffers
            mujoco.mjr_readPixels(self.rgb_buffer, self.depth_buffer, self.viewport, self.context)

            # Check for blank buffer (context issue)
            if np.max(self.rgb_buffer) == 0:
                # Don't warn on every frame
                if not hasattr(self, '_blank_buffer_warned'):
                    print(" Blank RGB buffer detected, context may need reinitialization")
                    self._blank_buffer_warned = True
                
                # Try one reinit
                if not hasattr(self, '_reinit_attempted'):
                    self._reinit_attempted = True
                    self._reinit_rendering()
                    
                    if self.context is not None:
                        # Retry render
                        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
                        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.rgb_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                        mujoco.mjr_render(self.viewport, self.scene, self.context)
                        mujoco.mjr_readPixels(self.rgb_buffer, self.depth_buffer, self.viewport, self.context)
            
            # Process RGB
            rgb_processed = np.flipud(self.rgb_buffer.copy()).astype(np.float32) / 255.0
            
            # Process Depth
            depth_processed = self._process_depth(np.flipud(self.depth_buffer.copy()))
            
            # Combine RGBD
            rgbd = np.zeros((self.resolution, self.resolution, 4), dtype=np.float32)
            rgbd[:, :, :3] = rgb_processed
            rgbd[:, :, 3] = depth_processed
            
            return rgbd.flatten()
            
        except Exception as e:
            # Don't spam errors
            if not hasattr(self, '_render_error_logged'):
                print(f" Camera render error: {e}")
                self._render_error_logged = True
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
        self._ensure_initialized()
        
        if self.rgb_camera_id < 0 or self.context is None:
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
            if not hasattr(self, '_rgb_error_logged'):
                print(f" RGB render error: {e}")
                self._rgb_error_logged = True
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
    
    def is_available(self) -> bool:
        """Check if camera is available and functional"""
        self._ensure_initialized()
        return (self.rgb_camera_id >= 0 and 
                self.scene is not None and 
                self.context is not None and
                self._initialized)
    
    def close(self):
        """Clean shutdown"""
        try:
            self.context = None
            self.scene = None
            self._initialized = False
        except Exception:
            pass