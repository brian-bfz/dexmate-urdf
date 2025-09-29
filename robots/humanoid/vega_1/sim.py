import sapien
import numpy as np

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
    # robot = loader.load("/home/brian/Sabrina/Robot_Example_Code/camera_cali/dexmate-urdf/robots/humanoid/vega_1/vega_no_effector.urdf")
    robot = loader.load("/home/brian/Sabrina/Robot_Example_Code/camera_cali/dexmate-urdf/robots/humanoid/vega_1/vega.urdf")
    # Print all active joints for reference
    for i, joint in enumerate(robot.get_active_joints()):
        print(f"{i} {joint.name}")
    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0])) #origin, identity rotation quaternion

    # Joint indices corresponding to vega.urdf file
    # Mapping from vega_no_effector.urdf indices to vega.urdf indices:
    # 0 B_wheel_j1 -> 0, 1 R_wheel_j1 -> 2, 2 L_wheel_j1 -> 4, 3 torso_j1 -> 6
    # 4 B_wheel_j2 -> 1, 5 R_wheel_j2 -> 3, 6 L_wheel_j2 -> 5, 7 torso_j2 -> 7
    # 8 torso_j3 -> 8, 9 head_j1 -> 10, 10 L_arm_j1 -> 16, 11 R_arm_j1 -> 25
    # 12 head_j2 -> 11, 13 L_arm_j2 -> 17, 14 R_arm_j2 -> 26, 15 head_j3 -> 12
    # 16 L_arm_j3 -> 18, 17 R_arm_j3 -> 27, 18 L_arm_j4 -> 19, 19 R_arm_j4 -> 28
    # 20 L_arm_j5 -> 20, 21 R_arm_j5 -> 29, 22 L_arm_j6 -> 21, 23 R_arm_j6 -> 30
    # 24 L_arm_j7 -> 22, 25 R_arm_j7 -> 31
    # active_joint_indices = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 16, 25, 11, 17, 26, 12, 18, 27, 19, 28, 20, 29, 21, 30, 22, 31]
    
    # # Get the actual joint objects for our selected joints, filtering out joints with empty names or 0 DOF
    # active_joints = []
    # for idx in active_joint_indices:
    #     joint = robot.get_joints()[idx]
    #     if joint.name and joint.get_dof() > 0:  # Only include joints with names and DOF > 0
    #         active_joints.append(joint)
    #     else:
    #         print(f"Skipping joint at index {idx}: name='{joint.name}', DOF={joint.get_dof()}")
    
    # Initialize qpos for all robot DOFs
    init_qpos = np.zeros(robot.dof)
    # one_qpos = init_qpos+1

    robot.set_qpos(init_qpos)
    # Hardcoded active_joints using joint indices from vega.urdf
    # These correspond to the same joints as in vega_no_effector.urdf
    # Mapping: B_wheel_j1, R_wheel_j1, L_wheel_j1, torso_j1, B_wheel_j2, R_wheel_j2, L_wheel_j2, 
    # torso_j2, torso_j3, head_j1, L_arm_j1, R_arm_j1, head_j2, L_arm_j2, R_arm_j2, head_j3, 
    # L_arm_j3, R_arm_j3, L_arm_j4, R_arm_j4, L_arm_j5, R_arm_j5, L_arm_j6, R_arm_j6, L_arm_j7, R_arm_j7
    # active_joint_indices = [1, 2, 3, 4, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39, 40]
    active_joint_indices = [1, 2, 3, 4, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39, 40]
    active_joints = [robot.get_joints()[idx] for idx in active_joint_indices]

    #lock_list = [1,2,5,6,9,12,15] #wheels and head



    for joint_idx, joint in enumerate(active_joints):
        print(joint_idx, joint.name)
        if "torso" in joint.name:
            joint.set_drive_property(stiffness=4000, damping=500, mode="acceleration")
        else:
            joint.set_drive_property(stiffness=4000, damping=500, force_limit=1000, mode="force")
        joint.set_drive_target(init_qpos[joint_idx])
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
