import sapien
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import transforms3d
import time
from kinematics_helper import KinHelper
from scipy.spatial.transform import Slerp, Rotation as R

class RightArmController:
    """
    A SAPIEN robot wrapper that includes forward kinematics functionality
    """
    
    def __init__(self, robot_name: str = "vega_no_effector", urdf_path: str = None):
        """
        Initialize the robot with SAPIEN and set up FK capabilities
        
        Args:
            urdf_path: Path to the URDF file
            scene: SAPIEN scene
            fix_root_link: Whether to fix the root link
        """
        if urdf_path is None:
            urdf_path = "vega_no_effector.urdf"

        self.kin_helper = self._create_kin_helper_with_custom_path(urdf_path)
        self.robot_name = robot_name

    def _create_kin_helper_with_custom_path(self, urdf_path: str):
        """Create a KinHelper instance with a custom URDF path"""
        # Import the necessary modules
        import urdfpy
        import sapien.core as sapien
        
        # Create a minimal KinHelper-like object
        class CustomKinHelper:
            def __init__(self, urdf_path):
                self.urdf_robot = urdfpy.URDF.load(urdf_path)
                # Create sapien robot
                self.engine = sapien.Engine()
                self.scene = self.engine.create_scene()
                loader = self.scene.create_urdf_loader()
                self.sapien_robot = loader.load(urdf_path)
                self.robot_model = self.sapien_robot.create_pinocchio_model()
                self.link_name_to_idx = {}
                for link_idx, link in enumerate(self.sapien_robot.get_links()):
                    self.link_name_to_idx[link.name] = link_idx
                self.sapien_eef_idx = None  # Set to None for now
            
            def compute_ik_from_mat(self, initial_qpos, tf_mat, damp=1e-1, eef_idx=None, active_qmask=None):
                """Compute IK given initial joint pos and target pose in matrix form"""
                pose = sapien.Pose(tf_mat)
                if active_qmask is None:
                    active_qmask = np.ones(len(initial_qpos), dtype=bool)
                
                qpos = self.robot_model.compute_inverse_kinematics(
                    link_index=eef_idx if eef_idx is not None else self.sapien_eef_idx,
                    pose=pose,
                    initial_qpos=initial_qpos,
                    active_qmask=active_qmask,
                    eps=1e-3,
                    damp=damp,
                )
                return qpos[0]
            
            def compute_fk_from_link_names(self, qpos, link_names):
                """Compute forward kinematics of robot links given joint positions"""
                self.robot_model.compute_forward_kinematics(qpos)
                link_idx_ls = [self.link_name_to_idx[link_name] for link_name in link_names]
                poses_ls = []
                for i in link_idx_ls:
                    pose = self.robot_model.get_link_pose(i)
                    poses_ls.append(pose.to_transformation_matrix())
                return {link_name: pose for link_name, pose in zip(link_names, poses_ls)}
            
            def compute_ik(self, initial_qpos, cartesian, **kwargs):
                """Compute inverse kinematics given initial joint pos and target pose"""
                tf_mat = np.eye(4)
                tf_mat[:3, :3] = transforms3d.euler.euler2mat(
                    ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes="sxyz"
                )
                tf_mat[:3, 3] = cartesian[0:3]
                return self.compute_ik_from_mat(
                    initial_qpos=initial_qpos,
                    tf_mat=tf_mat,
                    **kwargs
                )
        
        return CustomKinHelper(urdf_path)

    def compute_ik_from_mat(self, *args, **kwargs):
        """Compute IK given initial joint pos and target pose in matrix form"""
        return self.kin_helper.compute_ik_from_mat(*args, **kwargs)
    
    def compute_fk_from_link_names(self, *args, **kwargs):
        """Compute forward kinematics using KinHelper"""
        return self.kin_helper.compute_fk_from_link_names(*args, **kwargs)
    
    def compute_ik(self, *args, **kwargs):
        """Compute inverse kinematics using KinHelper"""
        return self.kin_helper.compute_ik(*args, **kwargs)



    
    def get_available_links(self) -> List[str]:
        """Get list of all available link names"""
        return list(self.link_name_to_idx.keys())
    
    def get_available_joints(self) -> List[str]:
        """Get list of all available joint names"""
        return list(self.joint_name_to_idx.keys())
    
    def set_joint_positions(self, qpos: np.ndarray):
        """Set robot joint positions"""
        self.robot.set_qpos(qpos)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        return self.robot.get_qpos()
    
    def add_link_offset(self, link_name: str, offset_matrix: np.ndarray):
        """Add a custom offset for a specific link"""
        self.offsets[link_name] = offset_matrix

    def extract_orientation_from_matrix(self, pose_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract orientation from 4x4 transformation matrix
        
        Args:
            pose_matrix: 4x4 transformation matrix
            
        Returns:
            Tuple of (rotation_matrix, euler_angles)
            - rotation_matrix: 3x3 rotation matrix
            - euler_angles: [roll, pitch, yaw] in radians
        """
        rotation_matrix = pose_matrix[:3, :3]
        
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        try:
            euler_angles = transforms3d.euler.mat2euler(rotation_matrix, axes='sxyz')
        except:
            # Fallback if conversion fails
            euler_angles = np.array([0.0, 0.0, 0.0])
        
        return rotation_matrix, euler_angles
    
    def compute_joint_orientations(self, qpos: np.ndarray, joint_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute orientation information for specified joints
        
        Args:
            qpos: Joint positions array
            joint_names: List of joint names to compute orientations for
            
        Returns:
            Dictionary mapping joint names to orientation information
        """
        # Set robot joint positions
        self.robot.set_qpos(qpos)
        self.scene.step()
        
        joint_orientations = {}
        
        for joint_name in joint_names:
            if joint_name in self.joint_name_to_idx:
                joint_idx = self.joint_name_to_idx[joint_name]
                joint = self.robot.get_active_joints()[joint_idx]
                
                # Get the child link of the joint (this gives us the joint's transformation)
                child_link = joint.get_child_link()
                if child_link is not None:
                    link_pose = child_link.get_pose()
                    
                    # Convert to transformation matrix
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = link_pose.to_transformation_matrix()[:3, :3]
                    pose_matrix[:3, 3] = link_pose.p
                    
                    # Extract orientation information
                    rotation_matrix, euler_angles = self.extract_orientation_from_matrix(pose_matrix)
                    
                    joint_orientations[joint_name] = {
                        'rotation_matrix': rotation_matrix,
                        'euler_angles': euler_angles,  # [roll, pitch, yaw] in radians
                        'quaternion': link_pose.q,  # SAPIEN quaternion [w, x, y, z]
                        'position': link_pose.p
                    }
                else:
                    # Fallback if no child link
                    joint_orientations[joint_name] = {
                        'rotation_matrix': np.eye(3),
                        'euler_angles': np.array([0.0, 0.0, 0.0]),
                        'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
                        'position': np.array([0.0, 0.0, 0.0])
                    }
            else:
                joint_orientations[joint_name] = {
                    'rotation_matrix': np.eye(3),
                    'euler_angles': np.array([0.0, 0.0, 0.0]),
                    'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
                    'position': np.array([0.0, 0.0, 0.0])
                }
        
        return joint_orientations
    
    
    def move_right_arm_joint_and_compute_fk(self, joint_name: str, angle_radians: float):
        """
        Move a specific right arm joint by the given angle and compute FK for all right arm links
        
        Args:
            joint_name: Name of the joint to move (e.g., 'R_arm_j1')
            angle_radians: Angle to move the joint in radians
            
        Returns:
            Dictionary mapping right arm link names to their 4x4 transformation matrices
        """
        # Angle is already in radians
        
        # Get current joint positions
        current_qpos = self.get_joint_positions()
        
        # Find the joint index
        if joint_name in self.joint_name_to_idx:
            joint_idx = self.joint_name_to_idx[joint_name]
            # Update the specific joint position
            current_qpos[joint_idx] += angle_radians
            #print(f"Moved {joint_name} by {angle_radians:.3f} rad (current position: {current_qpos[joint_idx]:.3f} rad)")
            # Set the new joint positions
            self.set_joint_positions(current_qpos)


            # Update the scene to reflect the new position
            self.scene.step()

            # Compute FK for all right arm links
            right_arm_links = ["R_arm_l1", "R_arm_l2", "R_arm_l3", "R_arm_l4", "R_arm_l5", "R_arm_l6", "R_arm_l7", "R_arm_l8"]
            available_links = self.get_available_links()
            target_links = [link for link in right_arm_links if link in available_links]
            
            
            fk_results = self.compute_fk_from_link_names(current_qpos, target_links)

            right_arm_joints = [joint.name for joint in self.robot.get_active_joints() if "R_arm" in joint.name]
            joint_orientations = self.compute_joint_orientations(current_qpos, right_arm_joints)

            
            print(f"Moved {joint_name} by {angle_radians:.3f} rad (current position: {current_qpos[joint_idx]:.3f} rad)")

            # Print link positions and orientations
            for link_name, pose_matrix in fk_results.items():
                position = pose_matrix[:3, 3]
                rotation_matrix, euler_angles = self.extract_orientation_from_matrix(pose_matrix)
                print(f"  {link_name}:")
                print(f"    Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                print(f"    Orientation (RPY): [{euler_angles[0]:.3f}, {euler_angles[1]:.3f}, {euler_angles[2]:.3f}] rad")
                print(f"    Orientation (RPY): [{np.degrees(euler_angles[0]):.1f}, {np.degrees(euler_angles[1]):.1f}, {np.degrees(euler_angles[2]):.1f}] deg")
            
            # Print joint orientations
            print(f"  Joint Orientations:")
            for joint_name, joint_info in joint_orientations.items():
                euler_angles = joint_info['euler_angles']
                quaternion = joint_info['quaternion']
                position = joint_info['position']
                print(f"    {joint_name}:")
                print(f"      Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                print(f"      Euler (RPY): [{euler_angles[0]:.3f}, {euler_angles[1]:.3f}, {euler_angles[2]:.3f}] rad")
                print(f"      Euler (RPY): [{np.degrees(euler_angles[0]):.1f}, {np.degrees(euler_angles[1]):.1f}, {np.degrees(euler_angles[2]):.1f}] deg")
                print(f"      Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")




            #for link_name, pose_matrix in fk_results.items():
                #position = pose_matrix[:3, 3]
                #print(f"  {link_name} position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            
            return fk_results
        else:
            print(f"Joint {joint_name} not found!")
            return {}

    def print_joint_orientations(self, qpos: np.ndarray, joint_names: List[str] = None):
        """
        Print detailed orientation information for specified joints
        
        Args:
            qpos: Joint positions array
            joint_names: List of joint names to analyze (if None, uses all right arm joints)
        """
        if joint_names is None:
            joint_names = [joint.name for joint in self.robot.get_active_joints() if "R_arm" in joint.name]
        
        joint_orientations = self.compute_joint_orientations(qpos, joint_names)
        
        print(f"\nJoint Orientations for {len(joint_names)} joints:")
        print("=" * 60)
        
        for joint_name, joint_info in joint_orientations.items():
            euler_angles = joint_info['euler_angles']
            quaternion = joint_info['quaternion']
            position = joint_info['position']
            rotation_matrix = joint_info['rotation_matrix']
            
            print(f"\n{joint_name}:")
            print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            print(f"  Euler Angles (RPY):")
            print(f"    Roll:  {euler_angles[0]:.3f} rad ({np.degrees(euler_angles[0]):.1f}°)")
            print(f"    Pitch: {euler_angles[1]:.3f} rad ({np.degrees(euler_angles[1]):.1f}°)")
            print(f"    Yaw:   {euler_angles[2]:.3f} rad ({np.degrees(euler_angles[2]):.1f}°)")
            print(f"  Quaternion (w,x,y,z): [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            print(f"  Rotation Matrix:")
            print(f"    [{rotation_matrix[0,0]:.3f}, {rotation_matrix[0,1]:.3f}, {rotation_matrix[0,2]:.3f}]")
            print(f"    [{rotation_matrix[1,0]:.3f}, {rotation_matrix[1,1]:.3f}, {rotation_matrix[1,2]:.3f}]")
            print(f"    [{rotation_matrix[2,0]:.3f}, {rotation_matrix[2,1]:.3f}, {rotation_matrix[2,2]:.3f}]")
    
    def sequential_right_arm_movement_demo(self, angle_radians: float = 0.175, delay_steps: int = 100, custom_angles: dict = None):
        """
        Sequentially move each right arm joint by the specified angle and compute FK
        
        Args:
            angle_radians: Default angle to move each joint in radians (used if custom_angles not provided)
            delay_steps: Number of simulation steps to wait between joint movements
            custom_angles: Dictionary mapping joint names to specific angles in radians
                          Example: {"R_arm_j1": 0.262, "R_arm_j2": 0.087, "R_arm_j3": 0.349}
        """
        # Get all right arm joint names
        right_arm_joints = [joint.name for joint in self.robot.get_active_joints() if "R_arm" in joint.name]
        right_arm_joints.sort()  # Sort to ensure consistent order
        
        print(f"Right arm joints found: {right_arm_joints}")
        
        # Set up angles for each joint
        if custom_angles:
            print("Using custom angles for each joint:")
            joint_angles = []
            for joint_name in right_arm_joints:
                angle = custom_angles.get(joint_name, angle_radians)
                joint_angles.append(angle)
                print(f"  {joint_name}: {angle:.3f} rad")
        else:
            print(f"Moving each joint by {angle_radians:.3f} rad sequentially...")
            joint_angles = [angle_radians] * len(right_arm_joints)
        
        # Initialize joint positions to zero
        #init_qpos = np.zeros(self.robot.dof)
        #self.set_joint_positions(init_qpos)
        
        step_count = 0
        joint_index = 0
        
        while not self.viewer.closed and joint_index < len(right_arm_joints):
            # Move to next joint every delay_steps
            # print(f"Step count: {step_count}") 
            if step_count % delay_steps == 0 and step_count > 0:
                joint_name = right_arm_joints[joint_index]
                joint_angle = joint_angles[joint_index]
                print(f"\n--- Moving {joint_name} by {joint_angle:.3f} rad ---")
                self.move_right_arm_joint_and_compute_fk(joint_name, joint_angle)
                joint_index += 1
            
            # Run physics simulation
            #for _ in range(4):  # render every 4 steps
                # Store current joint positions before physics step
                #current_positions = self.robot.get_qpos().copy()
                
                # Compute passive forces but don't apply them to avoid resetting joint positions
                # qf = self.robot.compute_passive_force(
                #     gravity=True,
                #     coriolis_and_centrifugal=True,
                # )
                # self.robot.set_qf(qf)  # This was causing the reset!
                
                #self.scene.step()
                
                # Restore joint positions after physics step to maintain composition
                #self.robot.set_qpos(current_positions)
            

            current_positions = self.robot.get_qpos().copy()
            self.scene.step()
            self.robot.set_qpos(current_positions)

            
            # Update rendering
            self.scene.update_render()
            self.viewer.render()
            step_count += 1

        
        print(f"\nCompleted sequential movement of {len(right_arm_joints)} right arm joints!")

def demo_with_ik(fix_root_link, balance_passive_force):
    """Demo function showing IK to reach the target pose from right_arm_4.0.py"""
    
    # Initialize robot with FK capabilities
    right_arm_controller = RightArmController("vega_no_effector")

    # Create SAPIEN scene
    scene = sapien.Scene()
    scene.add_ground(-0.1)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    
    # Create viewer
    viewer = scene.create_viewer()
    # Facing back of the Robot
    #viewer.set_camera_xyz(x=-2, y=0, z=1)
    #viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Facing the Robot from the Front
    #viewer.set_camera_xyz(x=2, y=0, z=1)  # Changed x from -2 to 2 to face robot from front
    #viewer.set_camera_rpy(r=0, p=-0.3, y=np.pi)  # Changed yaw from 0 to pi to face opposite direction

    # Facing the Robot from the Right
    viewer.set_camera_xyz(x=0, y=-2, z=1)  # Position camera on right side of robot
    viewer.set_camera_rpy(r=0, p=-0.3, y=-np.pi/2)  # Face robot from right side
    
    
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    robot = loader.load("/home/brian/Sabrina/Robot_Example_Code/vega-urdf/vega_no_effector.urdf")
    
    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])
    robot.set_root_pose(
        sapien.Pose([0, 0, 0], [1, 0, 0, 0])
    )
    # Initialize joint positions from joint_poses.txt file
    # Joint positions in radians from the actual robot state
    joint_positions = {
        # Left Arm (7 joints)
        'L_arm_j1': 1.577940,
        'L_arm_j2': 0.011170,
        'L_arm_j3': 0.000766,
        'L_arm_j4': -3.064725,
        'L_arm_j5': 0.000744,
        'L_arm_j6': 0.001414,
        'L_arm_j7': -0.697985,
        
        # Right Arm (7 joints)
        'R_arm_j1': 0.000023,
        'R_arm_j2': -0.000072,
        'R_arm_j3': 0.000096,
        'R_arm_j4': -0.000024,
        'R_arm_j5': 0.000024,
        'R_arm_j6': -0.000047,
        'R_arm_j7': 0.000073,
        
        # Torso (3 joints)
        'torso_j1': 1.043532,
        'torso_j2': 1.951278,
        'torso_j3': 0.522552,
        
        # Head (3 joints)
        'head_j1': -0.388510,
        'head_j2': -0.000559,
        'head_j3': -0.004765,
        
        # Chassis (6 joints)
        'L_wheel_j1': 0.000000,  # steer
        'R_wheel_j1': 0.000000,  # steer
        'B_wheel_j1': 0.000000,  # back wheel steer
        'L_wheel_j2': 0.000000,  # drive
        'R_wheel_j2': 0.000000,  # drive
        'B_wheel_j2': 0.000000,  # back wheel drive
    }
    
    # Create initial joint position array
    init_qpos = np.zeros(robot.dof)
    
    # Map joint names to their positions
    active_joints = robot.get_active_joints()
    for i, joint in enumerate(active_joints):
        joint_name = joint.name
        if joint_name in joint_positions:
            init_qpos[i] = joint_positions[joint_name]
            print(f"Set {joint_name} to {joint_positions[joint_name]:.6f} radians")
        else:
            print(f"Warning: Joint {joint_name} not found in joint_positions mapping")
    
    robot.set_qpos(init_qpos)
    # final_qpos = np.zeros(robot.dof)
    # final_qpos[10] = 1.57/2.0 # j1
    # final_qpos[11] = -1.57/2.0 # j1
    # final_qpos[18] = 1.57/2.0 # j4
    # final_qpos[19] = 1.57/2.0 # j4


    
    # Print available links and joints
    #print("Available links:", robot.get_available_links()[:10])  # Show first 10
    #print("Available joints:", robot.get_available_joints()[:10])  # Show first 10
    
    # Set up joint properties
    # active_joints = robot_fk.robot.get_active_joints()
    # for joint_idx, joint in enumerate(active_joints):
        # if "torso" in joint.name:
            #joint.set_drive_property(stiffness=4000, damping=500, mode="acceleration")
        # else:
            #joint.set_drive_property(stiffness=4000, damping=500, force_limit=1000, mode="force")
            #joint.set_drive_property(stiffness=100000, damping=1000, mode="position")
    

    active_joints = robot.get_active_joints()

    # Get initial pose
    init_right_arm_pose = (
        robot.get_links()[38].get_entity_pose().to_transformation_matrix().copy()
    )
    target_right_arm_pose = init_right_arm_pose.copy()
    #target_right_arm_pose[0, 3] += -0.400
    #target_right_arm_pose[1, 3] += -0.150
    target_right_arm_pose[2, 3] += 0.207

    alpha = 0.4
    beta = 0.4
    gamma = 0.2

    for joint_idx, joint in enumerate(active_joints):
        if "torso" in joint.name:
            joint.set_drive_property(stiffness=4000, damping=500, mode="acceleration")
        else:
            joint.set_drive_property(
                stiffness=4000, damping=500, force_limit=1000, mode="force"
            )
        joint.set_drive_target(init_qpos[joint_idx])
    
    # Initialize joint positions
    

    # Movement parameters
    movement_duration = 200  # Number of steps to complete the movement
    step_count = 0
    movement_complete = False
    
    # Apply final rotation using Euler angles for target pose
    rotation_matrix = transforms3d.euler.euler2mat(
        ai=gamma, aj=beta, ak=alpha, axes="sxyz"
    )
    target_right_arm_pose[:3, :3] = rotation_matrix
    
    # Define which joints are active for right arm IK
    eef_idx = 38  # R_arm_l8 end-effector link index (corrected from 33)
    active_qmask = np.zeros(robot.dof, dtype=bool)
    # Activate right arm joints (R_arm_j1 through R_arm_j7, skip R_arm_j8 which is fixed)
    active_qmask[[11, 14, 17, 19, 21, 23, 25]] = True  # Correct right arm joint indices
    
    print("Starting smooth movement to target position...")
    
    while not viewer.closed:
        if not movement_complete:
            # Calculate interpolation ratio (0 to 1)
            ratio = min(step_count / movement_duration, 1.0)
            
            # Create interpolated pose
            curr_right_arm_pose = target_right_arm_pose.copy()
            curr_right_arm_pose[:3, 3] = (
                init_right_arm_pose[:3, 3] * (1 - ratio) + target_right_arm_pose[:3, 3] * ratio
            )
            
            # Interpolate rotation using spherical linear interpolation
            
            # Convert rotation matrices to Rotation objects
            init_rot = R.from_matrix(init_right_arm_pose[:3, :3])
            target_rot = R.from_matrix(target_right_arm_pose[:3, :3])
            
            # Create Slerp object
            slerp = Slerp([0, 1], R.concatenate([init_rot, target_rot]))
            
            # Interpolate rotation
            curr_rot = slerp([ratio])
            curr_right_arm_pose[:3, :3] = curr_rot[0].as_matrix()
            
            # Get current joint positions
            current_qpos = robot.get_qpos()
            
            # Compute IK for right arm only
            right_arm_ik_result = right_arm_controller.compute_ik_from_mat(
                current_qpos, curr_right_arm_pose, eef_idx=eef_idx, active_qmask=active_qmask
            )

            # Only update the right arm joints, keep others unchanged
            curr_qpos = current_qpos.copy()
            right_arm_indices = [11, 14, 17, 19, 21, 23, 25]
            curr_qpos[right_arm_indices] = right_arm_ik_result[right_arm_indices]
            
            # Check if movement is complete
            if ratio >= 1.0:
                movement_complete = True
                print("Movement complete! Holding position...")
        else:
            # Movement is complete, just hold the current position
            curr_qpos = robot.get_qpos()
        
        # Set joint positions
        robot.set_qpos(curr_qpos)
        
        # Set drive targets to maintain stability for all joints
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_target(curr_qpos[joint_idx])
        
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
        viewer.render()
        
        step_count += 1

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    parser.parse_args()

    demo_with_ik(
        # fix_root_link=args.fix_root_link,
        # balance_passive_force=args.balance_passive_force,
        fix_root_link=True,
        balance_passive_force=True,
    )


if __name__ == "__main__":
    main()











