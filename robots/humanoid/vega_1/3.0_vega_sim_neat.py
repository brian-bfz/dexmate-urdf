import argparse
import json
import numpy as np
from PIL import Image, ImageColor
import sapien

# Simulation for Vega robot with stereo camera functionality
# This file loads the Vega robot and provides stereo camera image capture
# Uses ZED camera mounts from URDF for realistic camera positioning
# Controls: Press 'c' to capture stereo camera images, 'p' to take viewer screenshot

# Constants
ROBOT_URDF_PATH = "/home/brian/Sabrina/Robot_Example_Code/camera_cali/dexmate-urdf/robots/humanoid/vega_1/vega_no_effector.urdf"
JOINT_POSITIONS_FILE = "/home/brian/Sabrina/Robot_Example_Code/camera_cali/dexmate-urdf/robots/humanoid/vega_1/joint_positions.json"

# Camera specifications from ZED X Mini
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FX = 779.0416870117188  # Focal length in pixels (x-direction)
CAMERA_FY = 779.0416870117188  # Focal length in pixels (y-direction)
CAMERA_CX = 990.1013793945312  # Principal point x-coordinate
CAMERA_CY = 576.9609985351562  # Principal point y-coordinate
CAMERA_NEAR = 0.1
CAMERA_FAR = 100
STEREO_BASELINE = 0.0499  # meters

# Joint drive properties
TORSO_STIFFNESS = 4000
TORSO_DAMPING = 500
OTHER_STIFFNESS = 4000
OTHER_DAMPING = 500
OTHER_FORCE_LIMIT = 1000


def setup_scene():
    """Initialize the simulation scene with lighting and ground."""
    scene = sapien.Scene()
    scene.add_ground(-0.1)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    return scene


def setup_viewer(scene):
    """Create and configure the viewer."""
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    return viewer


def load_robot(scene, fix_root_link):
    """Load the robot from URDF and configure collision groups."""
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(ROBOT_URDF_PATH)

    # Configure collision groups
    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])

    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    return robot


def get_joint_positions():
    """Load initial joint positions from configuration file."""
    try:
        with open(JOINT_POSITIONS_FILE, 'r') as f:
            config = json.load(f)
            joint_positions = config['joint_positions']
            print(f"Loaded joint positions from {JOINT_POSITIONS_FILE}")
            return joint_positions
    except FileNotFoundError:
        print(f"Error: Joint positions file not found at {JOINT_POSITIONS_FILE}")
        print("Please ensure the joint_positions.json file exists in the correct location.")
        raise


def configure_robot_joints(robot, joint_positions):
    """Configure robot joints with initial positions and drive properties."""
    active_joints = robot.get_active_joints()

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

    # Configure drive properties
    for joint_idx, joint in enumerate(active_joints):
        if "torso" in joint.name:
            joint.set_drive_property(
                stiffness=TORSO_STIFFNESS, damping=TORSO_DAMPING, mode="acceleration"
            )
        else:
            joint.set_drive_property(
                stiffness=OTHER_STIFFNESS,
                damping=OTHER_DAMPING,
                force_limit=OTHER_FORCE_LIMIT,
                mode="force",
            )
        joint.set_drive_target(init_qpos[joint_idx])

    print(f"Robot loaded with {robot.dof} degrees of freedom")
    print(f"Number of active joints: {len(active_joints)}")
    print("Active joints:")
    for i, joint in enumerate(active_joints):
        print(f"  {i}: {joint.name}")

    return active_joints


def calculate_camera_fov():
    """Calculate camera field of view from intrinsic parameters."""
    v_fov_deg = 2 * np.arctan(CAMERA_HEIGHT / (2 * CAMERA_FY))
    h_fov_deg = 2 * np.arctan(CAMERA_WIDTH / (2 * CAMERA_FX))

    print(
        f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT} (aspect ratio: {CAMERA_WIDTH/CAMERA_HEIGHT:.3f})"
    )
    print(f"Camera FOV: {h_fov_deg:.2f}° x {v_fov_deg:.2f}° (horizontal x vertical)")
    print(f"Camera FOV: {v_fov_deg:.4f} x {v_fov_deg:.4f} radians")

    return v_fov_deg


def setup_cameras(scene, robot):
    """Setup stereo cameras attached to robot links."""
    v_fov_deg = calculate_camera_fov()

    # Create cameras
    left_camera = scene.add_camera(
        name="left_camera",
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fovy=v_fov_deg,
        near=CAMERA_NEAR,
        far=CAMERA_FAR,
    )

    depth_camera = scene.add_camera(
        name="depth_camera",
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fovy=v_fov_deg,
        near=CAMERA_NEAR,
        far=CAMERA_FAR,
    )

    # Find camera links
    zed_depth_link = None
    zed_left_link = None
    head_link = None

    for link in robot.get_links():
        if link.get_name() == "zed_depth_frame":
            zed_depth_link = link
        elif link.get_name() == "zed_left_camera":
            zed_left_link = link
        elif link.get_name() == "head_l3":
            head_link = link

    # Position cameras
    left_world_pose = zed_left_link.get_entity_pose()
    depth_world_pose = zed_depth_link.get_entity_pose()
    # head_quat = head_link.get_entity_pose().q

    left_camera.set_entity_pose(left_world_pose)
    depth_camera.set_entity_pose(depth_world_pose)

    # Print camera information
    print("Left camera intrinsic matrix\n", left_camera.get_intrinsic_matrix())
    print("depth camera intrinsic matrix\n", depth_camera.get_intrinsic_matrix())

    print("\n=== ZED X Mini Camera Intrinsic Parameters ===")
    print(f"Resolution: {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"Intrinsic matrix parameters:")
    print(f"  fx: {CAMERA_FX:.2f} pixels (focal length x-direction)")
    print(f"  fy: {CAMERA_FY:.2f} pixels (focal length y-direction)")
    print(f"  cx: {CAMERA_CX:.2f} pixels (principal point x)")
    print(f"  cy: {CAMERA_CY:.2f} pixels (principal point y)")
    print(
        f"Calculated Field of View: {2 * np.arctan(CAMERA_WIDTH / (2 * CAMERA_FX)):.2f}° x {v_fov_deg:.2f}° (H x V)"
    )
    print(f"Stereo baseline: {STEREO_BASELINE} m")
    print("===============================================")

    return left_camera, depth_camera


def capture_stereo_images(scene, left_camera, depth_camera):
    """Capture and save images from both cameras."""
    scene.step()
    scene.update_render()
    left_camera.take_picture()
    depth_camera.take_picture()

    # Capture color images
    left_rgba = left_camera.get_picture("Color")
    left_rgba_img = (left_rgba * 255).clip(0, 255).astype("uint8")
    Image.fromarray(left_rgba_img).save("left_color.png")

    depth_rgba = depth_camera.get_picture("Color")
    depth_rgba_img = (depth_rgba * 255).clip(0, 255).astype("uint8")
    Image.fromarray(depth_rgba_img).save("depth_color.png")

    # Capture depth images
    left_position = left_camera.get_picture("Position")
    left_depth = -left_position[..., 2]
    left_depth_image = (left_depth * 1000.0).astype(np.uint16)
    Image.fromarray(left_depth_image).save("left_depth.png")

    depth_position = depth_camera.get_picture("Position")
    depth_depth = -depth_position[..., 2]
    depth_depth_image = (depth_depth * 1000.0).astype(np.uint16)
    Image.fromarray(depth_depth_image).save("depth_depth.png")

    # Capture segmentation images
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array(
        [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
    )

    left_seg_labels = left_camera.get_picture("Segmentation")
    left_label0_image = left_seg_labels[..., 0].astype(np.uint8)
    left_label1_image = left_seg_labels[..., 1].astype(np.uint8)
    Image.fromarray(color_palette[left_label0_image]).save("left_label0.png")
    Image.fromarray(color_palette[left_label1_image]).save("left_label1.png")

    depth_seg_labels = depth_camera.get_picture("Segmentation")
    depth_label0_image = depth_seg_labels[..., 0].astype(np.uint8)
    depth_label1_image = depth_seg_labels[..., 1].astype(np.uint8)
    Image.fromarray(color_palette[depth_label0_image]).save("depth_label0.png")
    Image.fromarray(color_palette[depth_label1_image]).save("depth_label1.png")

    print("Stereo camera images saved:")
    print("  Left: left_color.png, left_depth.png, left_label0.png, left_label1.png")
    print(
        "  depth: depth_color.png, depth_depth.png, depth_label0.png, depth_label1.png"
    )


def demo(fix_root_link, balance_passive_force):
    """Main demo function for Vega robot simulation with stereo cameras."""
    # Setup scene and viewer
    scene = setup_scene()
    viewer = setup_viewer(scene)

    # Load and configure robot
    robot = load_robot(scene, fix_root_link)
    joint_positions = get_joint_positions()
    active_joints = configure_robot_joints(robot, joint_positions)

    # Setup cameras
    left_camera, depth_camera = setup_cameras(scene, robot)

    # Capture initial images
    capture_stereo_images(scene, left_camera, depth_camera)

    # Position viewer for better view
    head_link = None
    for link in robot.get_links():
        if link.get_name() == "head_l3":
            head_link = link
            break

    if head_link:
        head_world_pose = head_link.get_entity_pose()
        viewer_pos = head_world_pose.p + np.array([-0.2, 0, 0.1])
        viewer.set_camera_xyz(*viewer_pos)
        viewer.set_camera_rpy(0, -0.3, 0)

    # Main simulation loop
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

        # Handle keyboard input
        if viewer.window.key_down("c"):  # Capture stereo images
            capture_stereo_images(scene, left_camera, depth_camera)
        if viewer.window.key_down("p"):  # Take screenshot
            rgba = viewer.window.get_picture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            Image.fromarray(rgba_img).save("screenshot.png")
            print("Screenshot saved as screenshot.png")

        viewer.render()


def main():
    """Main entry point for the Vega robot simulation."""
    parser = argparse.ArgumentParser(
        description="Vega robot simulation with stereo cameras"
    )
    parser.add_argument(
        "--fix-root-link", action="store_true", help="Fix the root link of the robot"
    )
    parser.add_argument(
        "--balance-passive-force", action="store_true", help="Balance passive forces"
    )
    args = parser.parse_args()

    demo(
        fix_root_link=args.fix_root_link,
        balance_passive_force=args.balance_passive_force,
    )


if __name__ == "__main__":
    main()
