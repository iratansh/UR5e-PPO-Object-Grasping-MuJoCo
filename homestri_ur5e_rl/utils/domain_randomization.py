"""
Domain Randomization for UR5e Sim-to-Real Transfer
Integrates with homestri patterns for optimal real-world transfer
"""

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple
import random

class DomainRandomizer:
    """
    Comprehensive domain randomization for sim-to-real transfer
    Based on successful sim-to-real research patterns
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        randomize_joints: bool = True,
        randomize_materials: bool = True,
        randomize_lighting: bool = True,
        randomize_camera: bool = True,
        randomize_dynamics: bool = True,
        randomize_geometry: bool = False,  # More advanced
    ):
        self.model = model
        self.randomize_joints = randomize_joints
        self.randomize_materials = randomize_materials
        self.randomize_lighting = randomize_lighting
        self.randomize_camera = randomize_camera
        self.randomize_dynamics = randomize_dynamics
        self.randomize_geometry = randomize_geometry
        # Allow independently gating friction randomization separate from visual materials
        self.randomize_friction = True
        # Store last applied milestone settings
        self._milestone_params = {}

        # Store original values for reset
        self._store_original_values()

        # Define randomization ranges based on sim-to-real literature
        self.randomization_ranges = {
            # Dynamics
            'joint_damping': (0.8, 1.2),
            'joint_frictionloss': (0.5, 2.0),
            'actuator_gain': (0.9, 1.1),
            'actuator_bias': (0.95, 1.05),
            
            # Materials
            'object_mass': (0.05, 0.5),  # 50g to 500g
            'object_friction': (0.3, 2.0),
            'object_restitution': (0.0, 0.3),
            'table_friction': (0.5, 1.5),
            
            # Visual
            'light_ambient': (0.3, 0.8),
            'light_diffuse': (0.4, 1.2),
            'light_specular': (0.0, 0.5),
            'material_rgba': (0.7, 1.3),  # Multiplicative
            'material_emission': (0.0, 0.1),
            
            # Camera noise
            'camera_noise_std': (0.0, 0.03),
            'depth_noise_std': (0.0, 0.02),
            
            # Geometry (advanced)
            'object_size': (0.9, 1.1),
            'gripper_width': (0.98, 1.02),
        }
        
        print(f" Domain randomizer initialized")
        print(f"   Joint randomization: {self.randomize_joints}")
        print(f"   Material randomization: {self.randomize_materials}")
        print(f"   Lighting randomization: {self.randomize_lighting}")
        print(f"   Camera randomization: {self.randomize_camera}")

    def set_milestone_parameters(
        self,
        mass_range: tuple = (50, 50),
        color_randomization: bool = False,
        lighting_randomization: bool = False,
        friction_randomization: bool = False,
        objects: Optional[List[str]] = None,
    ) -> None:
        """Apply per-milestone domain randomization settings.
        mass_range is expected in grams; convert to kg. Other flags gate existing
        randomizations without requiring re-instantiation.
        """
        # Convert grams to kilograms for internal MuJoCo values
        try:
            g_min, g_max = mass_range
            kg_min = max(0.001, float(g_min) / 1000.0)
            kg_max = max(kg_min, float(g_max) / 1000.0)
        except Exception:
            # Fallback to a conservative default if malformed
            kg_min, kg_max = 0.05, 0.05
        # Update ranges for object mass to sample absolute masses, not multipliers
        self.randomization_ranges['object_mass'] = (kg_min, kg_max)
        # Gate visual and physics randomizations based on milestone
        self.randomize_materials = bool(color_randomization)
        self.randomize_lighting = bool(lighting_randomization)
        self.randomize_friction = bool(friction_randomization)
        # Store params for reference
        self._milestone_params = {
            'mass_range_kg': (kg_min, kg_max),
            'color_randomization': self.randomize_materials,
            'lighting_randomization': self.randomize_lighting,
            'friction_randomization': self.randomize_friction,
            'objects': objects or [],
        }
        print(
            f"📝 Domain randomizer milestone params: mass={kg_min:.3f}-{kg_max:.3f}kg, "
            f"color={'ON' if self.randomize_materials else 'OFF'}, "
            f"lighting={'ON' if self.randomize_lighting else 'OFF'}, "
            f"friction={'ON' if self.randomize_friction else 'OFF'}"
        )
        
    def _store_original_values(self):
        """Store original model values for reset"""
        self.original_values = {
            'joint_damping': self.model.dof_damping.copy(),
            'joint_frictionloss': self.model.dof_frictionloss.copy(),
            'actuator_gain': self.model.actuator_gainprm.copy(),
            'body_mass': self.model.body_mass.copy(),
            'geom_friction': self.model.geom_friction.copy(),
            'geom_size': self.model.geom_size.copy(),
            'mat_rgba': self.model.mat_rgba.copy(),
            'mat_emission': self.model.mat_emission.copy(),
            'light_ambient': self.model.light_ambient.copy(),
            'light_diffuse': self.model.light_diffuse.copy(),
            'light_specular': self.model.light_specular.copy(),
        }
        
    def randomize(self, data: Optional[mujoco.MjData] = None):
        """Apply all randomizations"""
        if self.randomize_joints:
            self._randomize_joint_properties()
            
        if self.randomize_dynamics:
            self._randomize_dynamics()
            
        if self.randomize_materials:
            self._randomize_materials()
            
        if self.randomize_lighting:
            self._randomize_lighting()
            
        if self.randomize_geometry:
            self._randomize_geometry()
            
        # Camera randomization happens during rendering
        
    def _randomize_joint_properties(self):
        """Randomize joint properties for real robot variation"""
        # Joint damping - affects joint stiffness and response
        damping_mult = np.random.uniform(*self.randomization_ranges['joint_damping'])
        self.model.dof_damping[:] = self.original_values['joint_damping'] * damping_mult
        
        # Joint friction - affects smoothness
        friction_mult = np.random.uniform(*self.randomization_ranges['joint_frictionloss'])
        self.model.dof_frictionloss[:] = self.original_values['joint_frictionloss'] * friction_mult
        
    def _randomize_dynamics(self):
        """Randomize actuator dynamics"""
        # Actuator gains - affects control response
        for i in range(self.model.nu):
            gain_mult = np.random.uniform(*self.randomization_ranges['actuator_gain'])
            self.model.actuator_gainprm[i, 0] = self.original_values['actuator_gain'][i, 0] * gain_mult
            
            # Small bias randomization
            if len(self.model.actuator_biasprm[i]) > 1:
                bias_mult = np.random.uniform(*self.randomization_ranges['actuator_bias'])
                self.model.actuator_biasprm[i, 1] *= bias_mult
                
    def _randomize_materials(self):
        """Randomize material properties for object variation"""
        # Object masses
        object_body_names = ["cube_object", "sphere_object", "cylinder_object"]
        
        for body_name in object_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    # Mass randomization (absolute mass in kg)
                    mass_val = np.random.uniform(*self.randomization_ranges['object_mass'])
                    self.model.body_mass[body_id] = mass_val
                    
            except:
                continue
                
        # Object friction
        object_geom_names = ["cube", "sphere", "cylinder"]
        
        if self.randomize_friction:
            for geom_name in object_geom_names:
                try:
                    geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                    if geom_id >= 0:
                        # Friction randomization
                        friction_mult = np.random.uniform(*self.randomization_ranges['object_friction'])
                        self.model.geom_friction[geom_id, 0] = 0.7 * friction_mult  # Sliding friction
                        self.model.geom_friction[geom_id, 1] = 0.005 * friction_mult  # Torsional friction
                        self.model.geom_friction[geom_id, 2] = 0.0001 * friction_mult  # Rolling friction
                        
                except:
                    continue
                
        # Table friction
        if self.randomize_friction:
            try:
                table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface")
                if table_geom_id >= 0:
                    table_friction_mult = np.random.uniform(*self.randomization_ranges['table_friction'])
                    self.model.geom_friction[table_geom_id, 0] = 1.0 * table_friction_mult
            except:
                pass
            
    def _randomize_lighting(self):
        """Randomize lighting for visual variation"""
        for light_id in range(self.model.nlight):
            # Ambient lighting
            ambient_mult = np.random.uniform(*self.randomization_ranges['light_ambient'])
            self.model.light_ambient[light_id] = self.original_values['light_ambient'][light_id] * ambient_mult
            
            # Diffuse lighting
            diffuse_mult = np.random.uniform(*self.randomization_ranges['light_diffuse'])
            self.model.light_diffuse[light_id] = self.original_values['light_diffuse'][light_id] * diffuse_mult
            
            # Specular lighting
            specular_mult = np.random.uniform(*self.randomization_ranges['light_specular'])
            self.model.light_specular[light_id] = self.original_values['light_specular'][light_id] * specular_mult
            
        # Material colors
        material_names = ["cube_mat", "sphere_mat", "cylinder_mat", "table_mat"]
        
        for mat_name in material_names:
            try:
                mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
                if mat_id >= 0:
                    # RGBA randomization (keep alpha constant)
                    for channel in range(3):  # RGB only
                        rgba_mult = np.random.uniform(*self.randomization_ranges['material_rgba'])
                        new_value = self.original_values['mat_rgba'][mat_id, channel] * rgba_mult
                        self.model.mat_rgba[mat_id, channel] = np.clip(new_value, 0.1, 1.0)
                        
                    # Emission randomization
                    emission_add = np.random.uniform(*self.randomization_ranges['material_emission'])
                    for channel in range(3):
                        self.model.mat_emission[mat_id, channel] = emission_add
                        
            except:
                continue
                
    def _randomize_geometry(self):
        """Randomize geometry (advanced feature)"""
        if not self.randomize_geometry:
            return
            
        # Object size randomization
        object_geom_names = ["cube", "sphere", "cylinder"]
        
        for geom_name in object_geom_names:
            try:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id >= 0:
                    size_mult = np.random.uniform(*self.randomization_ranges['object_size'])
                    
                    # Scale all size dimensions
                    for i in range(3):  # x, y, z
                        if self.model.geom_size[geom_id, i] > 0:
                            self.model.geom_size[geom_id, i] = self.original_values['geom_size'][geom_id, i] * size_mult
                            
            except:
                continue
                
    def add_camera_noise(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add realistic camera noise for sim-to-real"""
        if not self.randomize_camera:
            return rgb_image, depth_image
            
        # RGB noise
        rgb_noise_std = np.random.uniform(0, self.randomization_ranges['camera_noise_std'][1])
        rgb_noise = np.random.normal(0, rgb_noise_std, rgb_image.shape)
        rgb_noisy = np.clip(rgb_image + rgb_noise, 0, 1)
        
        # Depth noise
        depth_noise_std = np.random.uniform(0, self.randomization_ranges['depth_noise_std'][1])
        
        # Add depth-dependent noise (more noise at greater distances)
        depth_mask = depth_image > 0
        depth_noise = np.random.normal(0, depth_noise_std, depth_image.shape)
        
        # Scale noise by depth
        depth_scaled_noise = depth_noise * (depth_image + 0.1)  # Avoid division by zero
        depth_noisy = depth_image.copy()
        depth_noisy[depth_mask] += depth_scaled_noise[depth_mask]
        depth_noisy = np.clip(depth_noisy, 0, None)
        
        # Add occasional pixel dropouts (simulate sensor errors)
        if np.random.random() < 0.05:  # 5% chance
            dropout_mask = np.random.random(depth_image.shape) < 0.01  # 1% pixels
            depth_noisy[dropout_mask] = 0
            
        return rgb_noisy, depth_noisy
        
    def reset_to_defaults(self):
        """Reset all randomizations to original values"""
        self.model.dof_damping[:] = self.original_values['joint_damping']
        self.model.dof_frictionloss[:] = self.original_values['joint_frictionloss']
        self.model.actuator_gainprm[:] = self.original_values['actuator_gain']
        self.model.body_mass[:] = self.original_values['body_mass']
        self.model.geom_friction[:] = self.original_values['geom_friction']
        self.model.geom_size[:] = self.original_values['geom_size']
        self.model.mat_rgba[:] = self.original_values['mat_rgba']
        self.model.mat_emission[:] = self.original_values['mat_emission']
        self.model.light_ambient[:] = self.original_values['light_ambient']
        self.model.light_diffuse[:] = self.original_values['light_diffuse']
        self.model.light_specular[:] = self.original_values['light_specular']
        
    def get_randomization_info(self) -> Dict:
        """Get current randomization parameters for logging"""
        return {
            'joint_damping_range': self.randomization_ranges['joint_damping'],
            'object_mass_range': self.randomization_ranges['object_mass'],
            'object_friction_range': self.randomization_ranges['object_friction'],
            'light_ambient_range': self.randomization_ranges['light_ambient'],
            'camera_noise_std_range': self.randomization_ranges['camera_noise_std'],
        }

class CurriculumDomainRandomizer(DomainRandomizer):
    """
    Domain randomizer with curriculum learning
    Gradually increases randomization difficulty
    """
    
    def __init__(self, model: mujoco.MjModel, **kwargs):
        super().__init__(model, **kwargs)
        self.curriculum_level = 0.1
        self.base_ranges = self.randomization_ranges.copy()
        
    def set_curriculum_level(self, level: float):
        """Update curriculum level (0.0 = minimal randomization, 1.0 = full randomization)"""
        self.curriculum_level = np.clip(level, 0.0, 1.0)
        
        # Scale randomization ranges based on curriculum
        for param, (min_val, max_val) in self.base_ranges.items():
            if param in ['joint_damping', 'actuator_gain', 'actuator_bias']:
                # Conservative scaling for dynamics
                scale = 0.1 + 0.9 * self.curriculum_level
            elif param in ['object_mass', 'object_friction']:
                # Moderate scaling for object properties
                scale = 0.3 + 0.7 * self.curriculum_level
            else:
                # Full scaling for visual properties
                scale = self.curriculum_level
                
            # Apply scaling
            mid_point = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            scaled_range = half_range * scale
            
            self.randomization_ranges[param] = (
                mid_point - scaled_range,
                mid_point + scaled_range
            )
            
        print(f" Domain randomization curriculum updated: {self.curriculum_level:.2f}")

# Test and validation functions
def test_domain_randomization():
    """Test domain randomization with sample model"""
    print(" Testing Domain Randomization...")
    
    xml = """
    <mujoco model="test_domain_rand">
        <worldbody>
            <light pos="0 0 3"/>
            <geom name="floor" type="plane" size="2 2 0.1" material="floor_mat"/>
            <body name="cube_object" pos="0 0 1">
                <geom name="cube" type="box" size="0.1 0.1 0.1" material="cube_mat"/>
                <joint type="free"/>
            </body>
        </worldbody>
        
        <asset>
            <material name="floor_mat" rgba="0.5 0.5 0.5 1"/>
            <material name="cube_mat" rgba="1 0 0 1"/>
        </asset>
    </mujoco>
    """
    
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # Test basic randomizer
    randomizer = DomainRandomizer(model)
    
    print(" Testing basic randomization...")
    original_mass = model.body_mass[1]  # Cube mass
    original_rgba = model.mat_rgba[1].copy()  # Cube color
    
    # Apply randomization
    randomizer.randomize(data)
    
    new_mass = model.body_mass[1]
    new_rgba = model.mat_rgba[1]
    
    print(f"   Mass: {original_mass:.3f} → {new_mass:.3f}")
    print(f"   Color: {original_rgba} → {new_rgba}")
    
    # Test curriculum randomizer
    print("\n Testing curriculum randomization...")
    curriculum_randomizer = CurriculumDomainRandomizer(model)
    
    for level in [0.1, 0.5, 1.0]:
        curriculum_randomizer.set_curriculum_level(level)
        print(f"   Level {level}: Object mass range = {curriculum_randomizer.randomization_ranges['object_mass']}")
        
    # Test camera noise
    print("\n Testing camera noise...")
    rgb_dummy = np.random.random((64, 64, 3))
    depth_dummy = np.random.random((64, 64)) * 2.0
    
    rgb_noisy, depth_noisy = randomizer.add_camera_noise(rgb_dummy, depth_dummy)
    
    rgb_diff = np.mean(np.abs(rgb_noisy - rgb_dummy))
    depth_diff = np.mean(np.abs(depth_noisy - depth_dummy))
    
    print(f"   RGB noise: {rgb_diff:.4f}")
    print(f"   Depth noise: {depth_diff:.4f}")
    
    print("\n Domain randomization test completed!")

if __name__ == "__main__":
    test_domain_randomization()