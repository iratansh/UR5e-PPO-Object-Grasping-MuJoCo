import mujoco
import numpy as np
from typing import Optional, Dict, Union
import os
import cv2
import glfw

class MujocoRenderer:
    """Renderer for MuJoCo environments."""
    
    def __init__(
        self, 
        model: "mujoco._structs.MjModel", 
        data: "mujoco._structs.MjData", 
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        use_mac_compatible_viewer: bool = False,
    ):
        self.model = model
        self.data = data
        self.default_camera_config = default_camera_config or {}
        self.use_mac_compatible_viewer = use_mac_compatible_viewer
        self.viewer_name = "MuJoCo Viewer"
        self.native_viewer = None  # For persistent MuJoCo viewer
        self.window = None
        self.window_context = None
        
        # Camera switching state 
        self._env = None  # Reference to environment for camera switching
        self._current_camera_index = 0
        self._available_cameras = []
        
        # Mouse interaction state
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_button_left = False
        self.mouse_button_right = False
        self.mouse_button_middle = False
        
        self._init_opengl_context()
        
        # Initialize rendering context
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        
        # Set default camera configuration
        self._set_camera_config()
        
        # Viewport for rendering
        width = getattr(model.vis.global_, 'offwidth', 480)
        height = getattr(model.vis.global_, 'offheight', 480)
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        
        # Initialize camera list - NEW
        self._initialize_available_cameras()

    def set_environment_reference(self, env):
        """Set reference to environment for camera switching - NEW"""
        self._env = env
        
    def _initialize_available_cameras(self):
        """Initialize list of available cameras - NEW"""
        self._available_cameras = []
        for i in range(self.model.ncam):
            camera_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if camera_name:
                self._available_cameras.append((camera_name, i))
        
        # Add free camera as first option
        self._available_cameras.insert(0, ("free_camera", -1))
        print(f" Available cameras: {[cam[0] for cam in self._available_cameras]}")

    def _switch_camera(self, direction):
        """Switch to next/previous camera - NEW"""
        if not self._available_cameras:
            return
            
        self._current_camera_index = (self._current_camera_index + direction) % len(self._available_cameras)
        camera_name, camera_id = self._available_cameras[self._current_camera_index]
        
        print(f" Switched to camera: {camera_name} (ID: {camera_id})")
        
        # Update the environment's camera if possible
        if self._env and hasattr(self._env, 'camera_name') and hasattr(self._env, 'camera_id'):
            self._env.camera_name = camera_name if camera_name != "free_camera" else None
            self._env.camera_id = camera_id if camera_id != -1 else None
        
        # Update renderer camera immediately
        if camera_id == -1:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.fixedcamid = -1
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = camera_id
            
    def _print_camera_help(self):
        """Print camera controls help - NEW"""
        print("\n Camera Controls:")
        print("   '[' - Previous camera")
        print("   ']' - Next camera") 
        print("   'h' - Show this help")
        print("   'r' - Reset view")
        print("   'f' - Focus on origin")
        print("   ESC - Exit")
        print("   Current cameras:")
        for i, (name, cam_id) in enumerate(self._available_cameras):
            marker = "* " if i == self._current_camera_index else "   "
            camera_type = "Free Camera" if cam_id == -1 else f"Fixed Camera ID {cam_id}"
            print(f"   {marker}{name} ({camera_type})")
        print()

    def _init_opengl_context(self):
        """Initialize OpenGL context."""
        try:
            # Try to get OpenGL backend from environment
            backend = os.environ.get("MUJOCO_GL", "glfw")
            
            if backend == "glfw":
                from mujoco.glfw import GLContext
                self.gl_context = GLContext(480, 480)
            elif backend == "egl":
                from mujoco.egl import GLContext
                self.gl_context = GLContext(480, 480)
            elif backend == "osmesa":
                from mujoco.osmesa import GLContext
                self.gl_context = GLContext(480, 480)
            else:
                # Default to glfw
                from mujoco.glfw import GLContext
                self.gl_context = GLContext(480, 480)
                
            # Make context current
            if self.gl_context is not None:
                self.gl_context.make_current()
            
        except Exception as e:
            print(f"Warning: Could not initialize OpenGL context: {e}")
            self.gl_context = None
            
    def _set_camera_config(self):
        """Set camera configuration from default config."""
        # Initialize camera with sensible defaults
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        
        # Set lookat to center of model
        if hasattr(self.data, 'geom_xpos') and len(self.data.geom_xpos) > 0:
            for i in range(3):
                self.camera.lookat[i] = np.median(self.data.geom_xpos[:, i])
        else:
            self.camera.lookat[:] = [0, 0, 0]
            
        # Set distance based on model extent
        if hasattr(self.model, 'stat') and hasattr(self.model.stat, 'extent'):
            self.camera.distance = self.model.stat.extent
        else:
            self.camera.distance = 2.0
            
        # Apply user configuration
        if "distance" in self.default_camera_config:
            self.camera.distance = self.default_camera_config["distance"]
        if "azimuth" in self.default_camera_config:
            self.camera.azimuth = self.default_camera_config["azimuth"]
        if "elevation" in self.default_camera_config:
            self.camera.elevation = self.default_camera_config["elevation"]
        if "lookat" in self.default_camera_config:
            self.camera.lookat[:] = self.default_camera_config["lookat"]

    def _render_cv2(self, rgb_array: np.ndarray):
        """Render using OpenCV for better compatibility."""
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.viewer_name, bgr_array)
        cv2.waitKey(1)

    def _mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events."""
        if button == 0:  # Left mouse button
            self.mouse_button_left = (action == 1)  # 1 = press, 0 = release
        elif button == 1:  # Right mouse button  
            self.mouse_button_right = (action == 1)
        elif button == 2:  # Middle mouse button
            self.mouse_button_middle = (action == 1)

    def _mouse_move_callback(self, window, xpos, ypos):
        """Handle mouse movement for camera control."""
        import glfw
        
        # Get window size for proper scaling
        width, height = glfw.get_window_size(window)
        
        # Calculate mouse movement delta
        dx = xpos - self.mouse_last_x
        dy = ypos - self.mouse_last_y
        
        # Update camera based on mouse button state
        if self.mouse_button_left:
            # Left drag: rotate camera (like Unity orbit)
            self.camera.azimuth += dx * 0.5
            self.camera.elevation += dy * 0.5
            
            # Clamp elevation to prevent flipping
            self.camera.elevation = max(-89, min(89, self.camera.elevation))
            
        elif self.mouse_button_right:
            # Right drag: pan camera (translate lookat point)
            # Scale movement based on distance
            scale = self.camera.distance * 0.002
            
            # Calculate pan vectors in camera space
            forward = np.array([
                np.cos(np.radians(self.camera.azimuth)) * np.cos(np.radians(self.camera.elevation)),
                np.sin(np.radians(self.camera.azimuth)) * np.cos(np.radians(self.camera.elevation)),
                np.sin(np.radians(self.camera.elevation))
            ])
            
            right = np.array([
                -np.sin(np.radians(self.camera.azimuth)),
                np.cos(np.radians(self.camera.azimuth)),
                0
            ])
            
            up = np.cross(right, forward)
            
            # Apply pan movement
            pan_offset = (-dx * scale * right) + (dy * scale * up)
            self.camera.lookat[0] += pan_offset[0]
            self.camera.lookat[1] += pan_offset[1]
            self.camera.lookat[2] += pan_offset[2]
        
        # Update last mouse position
        self.mouse_last_x = xpos
        self.mouse_last_y = ypos

    def _scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for zooming."""
        # Zoom in/out by adjusting camera distance
        zoom_speed = 0.1
        self.camera.distance *= (1.0 - yoffset * zoom_speed)
        
        # Clamp distance to reasonable bounds
        self.camera.distance = max(0.1, min(20.0, self.camera.distance))

    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input for additional controls."""
        import glfw
        
        if action == glfw.PRESS or action == glfw.REPEAT:
            # Reset camera view with 'R' key
            if key == glfw.KEY_R:
                self._set_camera_config()  # Reset to default view
            
            # Focus on origin with 'F' key
            elif key == glfw.KEY_F:
                self.camera.lookat[:] = [0, 0, 0]
            
            # Quit with ESC
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
                
        # NEW: Camera switching functionality
        if action == glfw.PRESS:
            if key == glfw.KEY_LEFT_BRACKET:  # '[' key
                self._switch_camera(-1)
            elif key == glfw.KEY_RIGHT_BRACKET:  # ']' key
                self._switch_camera(1)
            elif key == glfw.KEY_H:  # 'h' key for help
                self._print_camera_help()

    def render(
        self,
        render_mode: Optional[str] = None, 
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        env=None  # Environment reference for camera switching
    ):
        """Render the scene."""
        if render_mode is None:
            return None
            
        #  Store environment reference for camera switching
        if env is not None:
            self._env = env
            
        # Handle camera selection
        if render_mode == "human":
            if camera_id is not None:
                if camera_id == -1:
                    self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.camera.fixedcamid = camera_id
            elif camera_name is not None:
                try:
                    cam_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                    )
                    if cam_id >= 0:
                        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        self.camera.fixedcamid = cam_id
                    else:
                        # Camera not found, use free camera
                        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                except:
                    self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            # If no specific camera requested, maintain current camera type
        elif camera_id is not None:
            if camera_id == -1:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.camera.fixedcamid = camera_id
        elif camera_name is not None:
            try:
                camera_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.camera.fixedcamid = camera_id
            except:
                # If camera name not found, use free camera
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                
        # Make sure OpenGL context is current
        if self.gl_context is not None:
            self.gl_context.make_current()

            # Update scene
            mujoco.mjv_updateScene(
                self.model, self.data, self.option, None, self.camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
            )

            if render_mode == "human":
                if self.use_mac_compatible_viewer:
                    # Get RGB array and render with cv2
                    rgb_array = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(rgb_array, None, self.viewport, self.context)
                    self._render_cv2(np.flipud(rgb_array))
                    return None
                else:
                    # Use GLFW-based viewer for macOS compatibility
                    if self.native_viewer is None:
                        try:
                            import glfw
                            import platform
                            
                            if not glfw.init():
                                raise Exception("Failed to initialize GLFW")
                            
                            # macOS specific window hints to force window to foreground
                            if platform.system() == "Darwin":
                                glfw.window_hint(glfw.FOCUSED, glfw.TRUE)
                                glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
                                glfw.window_hint(glfw.FLOATING, glfw.TRUE)
                            
                            self.window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
                            if not self.window:
                                glfw.terminate()
                                raise Exception("Failed to create GLFW window")
                            
                            glfw.make_context_current(self.window)
                            
                            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
                            glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
                            glfw.set_scroll_callback(self.window, self._scroll_callback)
                            glfw.set_key_callback(self.window, self._key_callback)
                            
                            # Initialize mouse position
                            xpos, ypos = glfw.get_cursor_pos(self.window)
                            self.mouse_last_x = xpos
                            self.mouse_last_y = ypos
                            
                            # Initialize OpenGL for the GLFW window
                            width, height = glfw.get_framebuffer_size(self.window)
                            self.window_viewport = mujoco.MjrRect(0, 0, width, height)
                            
                            self.window_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
                            
                            # macOS specific fixes to bring window to front
                            if platform.system() == "Darwin":
                                glfw.show_window(self.window)
                                glfw.focus_window(self.window)
                                glfw.request_window_attention(self.window)
                                
                                # Try to use Cocoa to bring to front
                                try:
                                    import subprocess
                                    subprocess.run(['osascript', '-e', 
                                                  'tell application "System Events" to set frontmost of first process whose name is "Python" to true'], 
                                                  check=False, capture_output=True)
                                except:
                                    pass
                            
                            self.native_viewer = True  # Mark as created
                            
                        except Exception as e:
                            print(f"Warning: Could not create GLFW viewer: {e}")
                            # Fallback to OpenCV viewer
                            rgb_array = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
                            mujoco.mjr_readPixels(rgb_array, None, self.viewport, self.context)
                            self._render_cv2(np.flipud(rgb_array))
                            return None
                    
                    # Render to the GLFW window
                    if self.native_viewer and hasattr(self, 'window') and self.window:
                        try:
                            import glfw
                            if not glfw.window_should_close(self.window):
                                # Make sure the window context is current
                                glfw.make_context_current(self.window)
                                
                                # Update window viewport size if changed
                                width, height = glfw.get_framebuffer_size(self.window)
                                self.window_viewport = mujoco.MjrRect(0, 0, width, height)
                                
                                # Set framebuffer for window rendering  
                                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.window_context)
                                
                                # Update scene with current data
                                mujoco.mjv_updateScene(
                                    self.model, self.data, self.option, None, self.camera,
                                    mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
                                )
                                
                                # Render scene to window
                                mujoco.mjr_render(self.window_viewport, self.scene, self.window_context)
                                
                                # Swap buffers and poll events
                                glfw.swap_buffers(self.window)
                                glfw.poll_events()
                            else:
                                # Window closed, clean up
                                glfw.destroy_window(self.window)
                                glfw.terminate()
                                self.native_viewer = None
                                self.window = None
                        except Exception as e:
                            print(f"Warning: GLFW rendering error: {e}")
                    return None

            elif render_mode in ["rgb_array", "depth_array"]:
                # Set framebuffer for offscreen rendering
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

                # Render to offscreen buffer
                mujoco.mjr_render(self.viewport, self.scene, self.context)

                if render_mode == "rgb_array":
                    rgb_array = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(rgb_array, None, self.viewport, self.context)
                    return np.flipud(rgb_array)

                elif render_mode == "depth_array":
                    depth_array = np.zeros((self.viewport.height, self.viewport.width), dtype=np.float32)
                    mujoco.mjr_readPixels(None, depth_array, self.viewport, self.context)
                    return np.flipud(depth_array)

        return None

    def close(self):
        """Close the renderer and clean up resources."""
        if self.use_mac_compatible_viewer:
            cv2.destroyAllWindows()

        # Close GLFW window if it exists
        if hasattr(self, 'window') and self.window is not None:
            try:
                import glfw
                glfw.destroy_window(self.window)
                glfw.terminate()
            except:
                pass
            self.window = None
            self.native_viewer = None

        if hasattr(self, 'gl_context') and self.gl_context is not None:
            try:
                self.gl_context.free()
            except:
                pass
            self.gl_context = None
            
        if hasattr(self, 'context'):
            self.context = None
        if hasattr(self, 'scene'):
            self.scene = None