"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler


def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = 'vega_no_effector.urdf'
    # load as a kinematic articulation
    asset = loader.load_kinematic(urdf_path)
    assert asset, 'URDF not loaded.'


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = 1920, 1080
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(58.72),  # Corrected ZED camera vertical FOV (90° horizontal with 16:9 aspect ratio)
        near=near,
        far=far,
    )
    # Position camera at the robot's head location
    # Based on the URDF, the head cameras are mounted on head_l3
    # Using the position from zed_left_camera_mount: (0.0365, 0.023, 0.0489)
    # and orientation: rpy="-1.57079 0 -1.57079"
    head_cam_pos = np.array([0.0365, 0.023, 0.0489])
    head_cam_rpy = np.array([-1.57079, 0, -1.57079])
    
    # Convert RPY to rotation matrix
    from transforms3d.euler import euler2mat
    rot_matrix = euler2mat(head_cam_rpy[0], head_cam_rpy[1], head_cam_rpy[2], 'sxyz')
    
    # Create pose for camera at head location
    head_pose = sapien.Pose(p=head_cam_pos, q=sapien.Pose.from_transformation_matrix(
        np.block([[rot_matrix, head_cam_pos.reshape(3, 1)], [0, 0, 0, 1]])
    ).q)
    
    camera.set_pose(head_pose)
    print('Intrinsic matrix\n', camera.get_intrinsic_matrix())

    # Get the head_l3 link to attach camera to
    head_link = None
    for link in asset.get_links():
        if link.get_name() == "head_l3":
            head_link = link
            break
    
    if head_link:
        camera.set_parent(parent=head_link, keep_pose=False)
        print("Camera mounted on head_l3 link")
    else:
        print("Warning: head_l3 link not found, using free camera")
        camera_mount_actor = scene.create_actor_builder().build_kinematic()
        camera.set_parent(parent=camera_mount_actor, keep_pose=False)
        camera_mount_actor.set_pose(head_pose)

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    # ---------------------------------------------------------------------------- #
    # RGBA
    # ---------------------------------------------------------------------------- #
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    # An alias is also provided
    # rgba = camera.get_color_rgba()  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    rgba_pil.save('color.png')

    # ---------------------------------------------------------------------------- #
    # XYZ position in the camera space
    # ---------------------------------------------------------------------------- #
    # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
    position = camera.get_float_texture('Position')  # [H, W, 4]

    # OpenGL/Blender: y up and -z forward
    points_opengl = position[..., :3][position[..., 3] < 1]
    points_color = rgba[position[..., 3] < 1][..., :3]
    # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
    # camera.get_model_matrix() must be called after scene.update_render()!
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

    # SAPIEN CAMERA: z up and x forward
    # points_camera = points_opengl[..., [2, 0, 1]] * [-1, -1, 1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(points_color)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, coord_frame])

    # Depth
    depth = -position[..., 2]
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    depth_pil.save('depth.png')

    # ---------------------------------------------------------------------------- #
    # Segmentation labels
    # ---------------------------------------------------------------------------- #
    # Each pixel is (visual_id, actor_id/link_id, 0, 0)
    # visual_id is the unique id of each visual shape
    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                             dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    # Or you can use aliases below
    # label0_image = camera.get_visual_segmentation()
    # label1_image = camera.get_actor_segmentation()
    label0_pil = Image.fromarray(color_palette[label0_image])
    label0_pil.save('label0.png')
    label1_pil = Image.fromarray(color_palette[label1_image])
    label1_pil.save('label1.png')

    # ---------------------------------------------------------------------------- #
    # Take picture from the viewer
    # ---------------------------------------------------------------------------- #
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    # Set viewer to show the robot's head camera perspective
    # Get the world pose of the head link to position viewer
    if head_link:
        head_world_pose = head_link.get_pose()
        # Position viewer slightly behind and above the head for better view
        viewer_pos = head_world_pose.p + np.array([-0.2, 0, 0.1])
        viewer.set_camera_xyz(*viewer_pos)
        # Look towards the front of the robot
        viewer.set_camera_rpy(0, -0.3, 0)  # slight downward angle
    else:
        # Fallback to camera position
        model_matrix = camera.get_model_matrix()
        model_matrix = model_matrix[:, [2, 0, 1, 3]] * np.array([-1, -1, 1, 1])
        rpy = mat2euler(model_matrix[:3, :3]) * np.array([1, -1, -1])
        viewer.set_camera_xyz(*model_matrix[0:3, 3])
        viewer.set_camera_rpy(*rpy)
    
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    while not viewer.closed:
        if viewer.window.key_down('p'):  # Press 'p' to take the screenshot
            rgba = viewer.window.get_float_texture('Color')
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save('screenshot.png')
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == '__main__':
    main()
