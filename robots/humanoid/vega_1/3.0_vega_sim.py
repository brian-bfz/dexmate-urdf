import sapien
import numpy as np

# Simulation for Vega robot with full effector (hands) - vega.urdf
# This file loads the complete Vega robot including hand/effector components

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
    robot = loader.load("/home/brian/Sabrina/Robot_Example_Code/camera_cali/dexmate-urdf/robots/humanoid/vega_1/vega.urdf")

    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0])) #origin, identity rotation quaternion

    init_qpos = np.zeros(robot.dof)
    one_qpos = init_qpos+1

    robot.set_qpos(init_qpos)
    active_joints = robot.get_active_joints()
    
    print(f"Robot loaded with {robot.dof} degrees of freedom")
    print(f"Number of active joints: {len(active_joints)}")
    print("Active joints:")
    for i, joint in enumerate(active_joints):
        print(f"  {i}: {joint.name}")
    
    # # Check initial joint positions
    # initial_qpos = robot.get_qpos()
    # print(f"Initial joint positions shape: {initial_qpos.shape}")
    # print(f"Initial joint positions: {initial_qpos}")


    for joint_idx, joint in enumerate(active_joints):
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
        fix_root_link=True,balance_passive_force=True
        )


if __name__ == "__main__":
    main()
