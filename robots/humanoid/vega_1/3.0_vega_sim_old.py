import sapien
import numpy as np
from PIL import Image, ImageColor
# import open3d as o3d  # Commented out - not used in script and not available for Python 3.13
from transforms3d.euler import mat2euler, euler2mat

# Simulation for Vega robot with full effector (hands) - vega.urdf
# This file loads the complete Vega robot including hand/effector components
# Integrated with stereo camera functionality for image capture and visualization
# Uses both left and right head cameras from URDF (zed_left_camera_mount and zed_right_camera_mount)
# Controls: Press 'c' to capture stereo camera images, 'p' to take viewer screenshot

def demo(fix_root_link, balance_passive_force):
    scene = sapien.Scene()
    scene.add_ground(-0.1)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    robot = loader.load("/home/brian/Sabrina/Robot_Example_Code/camera_cali/dexmate-urdf/robots/humanoid/vega_1/vega_no_effector.urdf")

    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0])) #origin, identity rotation quaternion

    # Get active joints first
    active_joints = robot.get_active_joints()
    
    # Initialize joint positions from joint_poses.txt file
    # Joint positions in radians from the actual robot state
    joint_positions = {
        # Left Arm (7 joints) - Updated from joint_poses.txt
        'L_arm_j1': 1.577940,
        'L_arm_j2': 0.011170,
        'L_arm_j3': 0.000766,
        'L_arm_j4': -3.064725,
        'L_arm_j5': 0.000744,
        'L_arm_j6': 0.001414,
        'L_arm_j7': -0.697985,
        
        # Right Arm (7 joints) - Updated from joint_poses.txt
        'R_arm_j1': 0.000023,
        'R_arm_j2': -0.000072,
        'R_arm_j3': 0.000096,
        'R_arm_j4': -0.000024,
        'R_arm_j5': 0.000024,
        'R_arm_j6': -0.000047,
        'R_arm_j7': 0.000073,
        
        # Torso (3 joints) - Updated from joint_poses.txt
        'torso_j1': 1.043532,
        'torso_j2': 1.951278,
        'torso_j3': 0.522552,
        
        # Head (3 joints) - Updated from joint_poses.txt
        'head_j1': -0.388510,
        'head_j2': -0.000559,
        'head_j3': -0.004765,
        
        # Chassis (6 joints) - Not in joint_poses.txt (kept at 0)
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
    for i, joint in enumerate(active_joints):
        joint_name = joint.name
        if joint_name in joint_positions:
            init_qpos[i] = joint_positions[joint_name]
            print(f"Set {joint_name} to {joint_positions[joint_name]:.6f} radians")
        else:
            print(f"Warning: Joint {joint_name} not found in joint_positions mapping")

    robot.set_qpos(init_qpos)
    
    print(f"Robot loaded with {robot.dof} degrees of freedom")
    print(f"Number of active joints: {len(active_joints)}")
    print("Active joints:")
    for i, joint in enumerate(active_joints):
        print(f"  {i}: {joint.name}")
    
    # # Check initial joint positions
    # initial_qpos = robot.get_qpos()
    # print(f"Initial joint positions shape: {initial_qpos.shape}")
    # print(f"Initial joint positions: {initial_qpos}")

    # ---------------------------------------------------------------------------- #
    # Stereo Camera Setup - Both Head Cameras from URDF
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    # Camera specifications from ZED X Mini intrinsic data
    width, height = 1920, 1080
    print(f"Camera resolution: {width}x{height} (aspect ratio: {width/height:.3f})")
    
    # ZED X Mini camera intrinsic parameters from intrinsic.txt
    # Left camera: fx=779.04, fy=779.04, cx=990.10, cy=576.96
    # Right camera: fx=779.04, fy=779.04, cx=990.10, cy=576.96
    # Resolution: 1920 x 1080 pixels
    
    # Camera intrinsic parameters
    fx = 779.0416870117188  # Focal length in pixels (x-direction)
    fy = 779.0416870117188  # Focal length in pixels (y-direction)
    cx = 990.1013793945312  # Principal point x-coordinate
    cy = 576.9609985351562  # Principal point y-coordinate
    
    # Calculate FOV using formula: FOV = 2 * arctan(sensor_dimension / (2 * focal_length))
    # For vertical FOV: FOV_y = 2 * arctan(height / (2 * fy))
    # For horizontal FOV: FOV_x = 2 * arctan(width / (2 * fx))
    v_fov_deg = 2 * np.arctan(height / (2 * fy)) 
    h_fov_deg = 2 * np.arctan(width / (2 * fx)) 
    
    # Convert FOV to radians for Sapien
    # fovy_rad = np.deg2rad(v_fov_deg)
    # fovx_rad = np.deg2rad(h_fov_deg)
    
    print(f"Camera FOV: {h_fov_deg:.2f}° x {v_fov_deg:.2f}° (horizontal x vertical)")
    print(f"Camera FOV: {v_fov_deg:.4f} x {v_fov_deg:.4f} radians")
    
    # Create left camera (zed_left_camera_mount from URDF)
    left_camera = scene.add_camera(
        name="left_camera",
        width=width,
        height=height,
        fovy=v_fov_deg,  # Vertical FOV from ZED X Mini intrinsic data
        near=near,
        far=far,
    )
    
    # Create right camera (zed_right_camera_mount from URDF)
    right_camera = scene.add_camera(
        name="right_camera",
        width=width,
        height=height,
        fovy=v_fov_deg,  # Vertical FOV from ZED X Mini intrinsic data
        near=near,
        far=far,
    )
    
    # Left camera position from zed_left_camera_mount: (0.0365, 0.023, 0.0489)
    # left_cam_pos = np.array([0.0365, 0.023, 0.0489]) #ZED_left_mount
    left_cam_pos = np.array([0.025, 0.023, 0.0489]) #ZED_left_depth_mount
    # Right camera position from zed_right_camera_mount: (0.0365, -0.027, 0.0489)
    right_cam_pos = np.array([0.0365, -0.027, 0.0489])
    
    # Get the head_l3 link to calculate camera orientation toward end effector
    head_link = None
    left_ee_link = None
    right_ee_link = None
    # New: also get zed_depth_frame link to read pose directly
    zed_depth_link = None
    
    for link in robot.get_links():
        print(f"link: {link.get_name()}")
        if link.get_name() == "zed_left_camera":
            zed_depth_link = link
        elif link.get_name() == "head_l3":
            head_link = link
        elif link.get_name() == "L_ee":
            left_ee_link = link
        elif link.get_name() == "R_ee":
            right_ee_link = link
    
    print(f"zed_depth_link: {zed_depth_link}")
    print(f"head_link: {head_link}")
    print(f"left_ee_link: {left_ee_link}")
    print(f"right_ee_link: {right_ee_link}")
            
    
    # Calculate camera orientations to look toward end effectors
    def calculate_camera_orientation(cam_pos, target_pos):
        """Calculate camera orientation to look at target position"""
        # Calculate direction vector from camera to target
        direction = target_pos - cam_pos
        direction = direction / np.linalg.norm(direction)
        
        # Camera forward direction (x-axis in camera frame)
        forward = direction
        
        # Camera up direction (z-axis in camera frame) - pointing up
        up = np.array([0, 0, 1])
        
        # Camera right direction (y-axis in camera frame)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recalculate up to ensure orthogonality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix
        rot_matrix = np.column_stack([forward, right, up])
        
        # Convert to quaternion
        from transforms3d.quaternions import mat2quat
        quat = mat2quat(rot_matrix)
        
        return quat
    
    # Use fixed camera orientation for both cameras
    # cam_rpy = np.array([0, 0, -1.57079 + np.pi/2])  # Fixed orientation rotated 90° left on x-y plane, no roll
    cam_rpy = np.array([0, 0, 0])
    rot_matrix = euler2mat(cam_rpy[0], cam_rpy[1], cam_rpy[2], 'sxyz')
    from transforms3d.quaternions import mat2quat
    left_quat = mat2quat(rot_matrix)
    right_quat = mat2quat(rot_matrix)
    
    # Using zed_depth_frame link pose directly instead of head chain
    if zed_depth_link:
        zed_world_pose = zed_depth_link.get_entity_pose()
        zed_world_pos = zed_world_pose.p
        zed_world_quat = zed_world_pose.q
        # import pdb; pdb.set_trace()
        print(f"[ZED] zed_depth_frame world position: {zed_world_pos}")
        # from transforms3d.euler import quat2euler
        # zed_rpy = quat2euler(zed_world_quat, axes='sxyz')
        # print(f"[ZED] zed_depth_frame world rpy: {zed_rpy}")
        print(f"[ZED] zed_depth_frame world quaternion: {zed_world_quat}")
    else:
        # Fallback to previous logic if zed link missing
        if head_link:
            # Get head world pose (position + orientation)
            head_world_pose = head_link.get_entity_pose()
            head_world_pos = head_world_pose.p
            head_world_quat = head_world_pose.q
            print(f"[HEAD] Head world position: {head_world_pos}")
            # Previous implementation (position-only, fixed orientation)
            # head_world_pos = head_link.get_entity_pose().p
            # left_cam_world_pos = head_world_pos + left_cam_pos
            # right_cam_world_pos = head_world_pos + right_cam_pos
            
            # Rotate local camera offsets by head orientation to obtain world offsets
            from transforms3d.quaternions import quat2mat
            head_rot_matrix = quat2mat(head_world_quat)
            left_offset_world = head_rot_matrix.dot(left_cam_pos)
            right_offset_world = head_rot_matrix.dot(right_cam_pos)
            
            # Calculate camera world positions (position + rotated offset)
            left_cam_world_pos = head_world_pos + left_offset_world
            right_cam_world_pos = head_world_pos + right_offset_world
            
            print(f"Left camera world position: {left_cam_world_pos}")
            print(f"Right camera world position: {right_cam_world_pos}")
        else:
            print("Warning: head_l3 link not found, using free cameras")
    
    # Create poses for both cameras (using local coordinates relative to head_l3)
    left_pose = sapien.Pose(p=left_cam_pos, q=left_quat)
    right_pose = sapien.Pose(p=right_cam_pos, q=right_quat)
    
    # Set camera poses using zed_depth_frame world pose directly
    if zed_depth_link:
        print(f"[ZED] zed_depth_frame world position: {zed_world_pos}")
        print(f"[ZED] zed_depth_frame world quaternion: {zed_world_quat}")
        # left_world_pose = sapien.Pose(p=zed_world_pos, q=zed_world_quat)
        # right_world_pose = sapien.Pose(p=zed_world_pos, q=zed_world_quat)
        left_world_pose = sapien.Pose(p=zed_world_pos, q=head_link.get_entity_pose().q)
        right_world_pose = zed_world_pose
        left_camera.set_entity_pose(left_world_pose)
        right_camera.set_entity_pose(right_world_pose)
    else:
        # Fallback: use previous head-based or local coordinates if zed link not found
        if head_link:
            # Calculate world positions (with head orientation) and set entity poses
            head_world_pose = head_link.get_entity_pose()
            head_world_pos = head_world_pose.p
            head_world_quat = head_world_pose.q
            
            from transforms3d.quaternions import quat2mat, qmult
            head_rot_matrix = quat2mat(head_world_quat)
            left_offset_world = head_rot_matrix.dot(left_cam_pos)
            right_offset_world = head_rot_matrix.dot(right_cam_pos)
            
            left_cam_world_pos = head_world_pos + left_offset_world
            right_cam_world_pos = head_world_pos + right_offset_world
            
            left_world_quat = qmult(head_world_quat, left_quat)
            right_world_quat = qmult(head_world_quat, right_quat)
            
            left_world_pose = sapien.Pose(p=left_cam_world_pos, q=left_world_quat)
            right_world_pose = sapien.Pose(p=right_cam_world_pos, q=right_world_quat)
            
            left_camera.set_entity_pose(left_world_pose)
            right_camera.set_entity_pose(right_world_pose)
        else:
            # Fallback: use world coordinates if neither zed nor head link found
            left_camera.set_entity_pose(left_pose)
            right_camera.set_entity_pose(right_pose)
    
    # Print camera intrinsic matrices
    print('Left camera intrinsic matrix\n', left_camera.get_intrinsic_matrix())
    print('Right camera intrinsic matrix\n', right_camera.get_intrinsic_matrix())
    
    # Print ZED X Mini intrinsic parameters for reference
    print("\n=== ZED X Mini Camera Intrinsic Parameters ===")
    print(f"Resolution: {width} x {height}")
    print(f"Intrinsic matrix parameters:")
    print(f"  fx: {fx:.2f} pixels (focal length x-direction)")
    print(f"  fy: {fy:.2f} pixels (focal length y-direction)")
    print(f"  cx: {cx:.2f} pixels (principal point x)")
    print(f"  cy: {cy:.2f} pixels (principal point y)")
    print(f"Calculated Field of View: {h_fov_deg:.2f}° x {v_fov_deg:.2f}° (H x V)")
    print(f"Stereo baseline: 0.0499 m")
    print("===============================================")

    if head_link:
        print("Stereo cameras mounted on head_l3 link")
        print(f"Left camera position: {left_cam_pos}")
        print(f"Right camera position: {right_cam_pos}")
    else:
        print("Warning: head_l3 link not found, using free cameras")


    for joint_idx, joint in enumerate(active_joints):
        if "torso" in joint.name:
            joint.set_drive_property(stiffness=4000, damping=500, mode="acceleration")
        else:
            joint.set_drive_property(stiffness=4000, damping=500, force_limit=1000, mode="force")
        joint.set_drive_target(init_qpos[joint_idx])

    # ---------------------------------------------------------------------------- #
    # Stereo Camera Image Capture Functions
    # ---------------------------------------------------------------------------- #
    def capture_stereo_camera_images():
        """Capture and save images from both left and right cameras"""
        scene.step()  # make everything set
        scene.update_render()
        left_camera.take_picture()
        right_camera.take_picture()

        # Left Camera Images
        left_rgba = left_camera.get_picture('Color')  # [H, W, 4]
        left_rgba_img = (left_rgba * 255).clip(0, 255).astype("uint8")
        left_rgba_pil = Image.fromarray(left_rgba_img)
        left_rgba_pil.save('left_color.png')

        # Right Camera Images
        right_rgba = right_camera.get_picture('Color')  # [H, W, 4]
        right_rgba_img = (right_rgba * 255).clip(0, 255).astype("uint8")
        right_rgba_pil = Image.fromarray(right_rgba_img)
        right_rgba_pil.save('right_color.png')

        # Left Camera Depth
        left_position = left_camera.get_picture('Position')  # [H, W, 4]
        left_depth = -left_position[..., 2]
        left_depth_image = (left_depth * 1000.0).astype(np.uint16)
        left_depth_pil = Image.fromarray(left_depth_image)
        left_depth_pil.save('left_depth.png')

        # Right Camera Depth
        right_position = right_camera.get_picture('Position')  # [H, W, 4]
        right_depth = -right_position[..., 2]
        right_depth_image = (right_depth * 1000.0).astype(np.uint16)
        right_depth_pil = Image.fromarray(right_depth_image)
        right_depth_pil.save('right_depth.png')

        # Left Camera Segmentation
        left_seg_labels = left_camera.get_picture('Segmentation')  # [H, W, 4]
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                 dtype=np.uint8)
        left_label0_image = left_seg_labels[..., 0].astype(np.uint8)  # mesh-level
        left_label1_image = left_seg_labels[..., 1].astype(np.uint8)  # actor-level
        left_label0_pil = Image.fromarray(color_palette[left_label0_image])
        left_label0_pil.save('left_label0.png')
        left_label1_pil = Image.fromarray(color_palette[left_label1_image])
        left_label1_pil.save('left_label1.png')

        # Right Camera Segmentation
        right_seg_labels = right_camera.get_picture('Segmentation')  # [H, W, 4]
        right_label0_image = right_seg_labels[..., 0].astype(np.uint8)  # mesh-level
        right_label1_image = right_seg_labels[..., 1].astype(np.uint8)  # actor-level
        right_label0_pil = Image.fromarray(color_palette[right_label0_image])
        right_label0_pil.save('right_label0.png')
        right_label1_pil = Image.fromarray(color_palette[right_label1_image])
        right_label1_pil.save('right_label1.png')

        print("Stereo camera images saved:")
        print("  Left: left_color.png, left_depth.png, left_label0.png, left_label1.png")
        print("  Right: right_color.png, right_depth.png, right_label0.png, right_label1.png")

    # Capture initial stereo camera images
    capture_stereo_camera_images()

    # Set viewer to show the robot's head camera perspective
    if head_link:
        head_world_pose = head_link.get_entity_pose()
        # Position viewer slightly behind and above the head for better view
        viewer_pos = head_world_pose.p + np.array([-0.2, 0, 0.1])
        viewer.set_camera_xyz(*viewer_pos)
        # Look towards the front of the robot
        viewer.set_camera_rpy(0, -0.3, 0)  # slight downward angle
    else:
        # Fallback to left camera position
        model_matrix = left_camera.get_model_matrix()
        model_matrix = model_matrix[:, [2, 0, 1, 3]] * np.array([-1, -1, 1, 1])
        rpy = mat2euler(model_matrix[:3, :3]) * np.array([1, -1, -1])
        viewer.set_camera_xyz(*model_matrix[0:3, 3])
        viewer.set_camera_rpy(*rpy)

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
        
        # Check for keyboard input to capture stereo camera images
        if viewer.window.key_down('c'):  # Press 'c' to capture stereo camera images
            capture_stereo_camera_images()
        if viewer.window.key_down('p'):  # Press 'p' to take screenshot from viewer
            rgba = viewer.window.get_picture('Color')
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save('screenshot.png')
            print("Screenshot saved as screenshot.png")
            
        viewer.render()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    args = parser.parse_args()

    demo(
        #fix_root_link=args.fix_root_link,
        #balance_passive_force=args.balance_passive_force,
        fix_root_link=True, balance_passive_force=True
        )


if __name__ == "__main__":
    main()
