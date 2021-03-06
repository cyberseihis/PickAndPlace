import numpy as np
import dartpy as dp
import RobotDART as rd

from utils import create_grid, box_into_basket
from lab_utils import AdT
from enum import Enum
from functools import partial

dt = 0.001
simulation_time = 200.0
total_steps = int(simulation_time / dt)

# Create robot
packages = [("tiago_description", "tiago/tiago_description")]
robot = rd.Tiago(int(1. / dt), "tiago/tiago_steel.urdf", packages)

arm_dofs = [
        "arm_1_joint",
        "arm_2_joint",
        "arm_3_joint",
        "arm_4_joint",
        "arm_5_joint",
        "arm_6_joint",
        "arm_7_joint",
        "gripper_finger_joint",
        "gripper_right_finger_joint"]
robot.set_positions(
    np.array(
        [np.pi / 2., np.pi / 4., 0., np.pi / 2., 0., 0., np.pi / 2., 0.03,
         0.03]),
    arm_dofs)

# Control base - we make the base fully controllable
robot.set_actuator_type("servo", "rootJoint", False, True, False)
robot.set_commands([1.0, 0., 0.],
                   ['rootJoint_rot_z', 'rootJoint_pos_x', 'rootJoint_pos_y'])

# Create position grid for the box/basket
basket_positions, box_positions = create_grid()

# Create box
box_size = [0.04, 0.04, 0.04]
# Random cube position
box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0],
            box_positions[box_pt][1], box_size[2] / 2.0]
# box_pose = [0, 0, 0, 0, -1, 0.02]
box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                          0.9, 0.1, 0.1, 1.0], "box_" + str(0))

# Create basket
basket_packages = [("basket", "models/basket")]
basket = rd.Robot("models/basket/basket.urdf", basket_packages, "basket")
# Random basket position
basket_pt = np.random.choice(len(basket_positions))
basket_z_angle = 0.
basket_pose = [
        0.,
        0.,
        basket_z_angle,
        basket_positions[basket_pt][0],
        basket_positions[basket_pt][1],
        0.0008]
basket.set_positions(basket_pose)
basket.fix_to_world()


# Create Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 1280
gconfig.height = 960
graphics = rd.gui.Graphics(gconfig)

# Create simulator object
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("bullet")
simu.set_control_freq(100)
simu.set_graphics(graphics)
graphics.look_at((0., 4.5, 2.5), (0., 0., 0.25))
simu.add_checkerboard_floor()
simu.add_robot(robot)
simu.add_robot(box)
simu.add_robot(basket)

finish_counter = 0


class bt_condition(Enum):
    Success = True
    Failure = False


def rpose():
    rbp = robot.base_pose()
    rbt = rbp.translation()
    return rbt


def bpose():
    return box.base_pose().translation()


def arm_pose():
    arm_tf = robot.body_pose(end_ef)
    return arm_tf.translation()


def dist_to_box():
    return np.linalg.norm(bpose() - arm_pose())


def dist_to_basket():
    return np.linalg.norm(above_basket(0.4).translation() - arm_pose())


def horiz_to_basket():
    return np.linalg.norm(above_basket(0).translation()[:1] - arm_pose()[:1])


def delta_rot(target):
    t_pose = target.translation()
    vec = t_pose - rpose()
    radt = np.arctan2(vec[0], vec[1])
    return -radt + (np.pi / 2)


def angdif(x, y):
    sd = np.sin(x-y)
    cd = np.cos(x-y)
    return np.arctan2(sd, cd)


def angle_err(target):
    dero = delta_rot(target)
    rbr = robot.base_pose().rotation()
    ebrz = dp.math.matrixToEulerXYZ(rbr)[2]
    difz = angdif(dero, ebrz)
    return np.abs(difz)


def error_task(source, target):
    return rd.math.logMap(source.inverse().multiply(target))


def P_vel(source, target, Kp):
    Ad = AdT(source)
    pvel = Ad @ error_task(source, target)
    return pvel


end_ef = "gripper_grasping_frame"
mask_cmd = np.concatenate([np.zeros(2), np.ones(3),
                           np.zeros(12), np.ones(7), np.zeros(4)])

mask_rot = np.zeros(28)
mask_rot[2] = 1

grip_names = ['gripper_finger_joint']
open_palm = [1]
iron_fist = [-100]


def grab_distance():
    d = 0.025 if bpose()[2] <= 0.08 else 0.05
    return d


def death_grips(grab_cond: bool):
    grip_cmd = open_palm if grab_cond else iron_fist
    robot.set_commands(grip_cmd, grip_names)


def above_basket(basket_target_height):
    bas_pose = basket.base_pose()
    basket_up = bas_pose.translation() + [0, 0, basket_target_height]
    bas_pose.set_translation(basket_up)
    return bas_pose


def box_above_basket():
    return bt_condition(horiz_to_basket() < 0.05)


def cube_in_grasp() -> bool:
    cond = dist_to_box() <= grab_distance()
    return bt_condition(cond)


def Behavior_Tree():
    node_1 = partial(Fallback, [cube_in_grasp, pick_cube])
    node_2 = partial(Fallback, [box_above_basket, place_above_basket])
    node_3 = partial(Sequence, [node_2, place_basket])
    node_root = partial(Sequence, [node_1, node_3])
    node_root()


def Sequence(behaviors: list):
    for b in behaviors:
        if b() != bt_condition.Success:
            return bt_condition.Failure
    return bt_condition.Success


def Fallback(behaviors: list):
    for b in behaviors:
        if b() == bt_condition.Success:
            return bt_condition.Success
    return bt_condition.Failure


def pick_cube():
    box_tf = box.base_pose()
    task_controller(box_tf, True)


def place_above_basket():
    task_controller(above_basket(0.4), False)


def place_basket():
    task_controller(above_basket(0.), False)


def task_controller(target, grab_cond):
    arm_tf = robot.body_pose(end_ef)
    hand_rot = arm_tf.rotation()
    target.set_rotation(hand_rot)
    vel = P_vel(arm_tf, target, 1.)
    print(dist_to_box())
    my_mask = mask_rot if angle_err(target) > np.pi / 10 else mask_cmd
    jac = robot.jacobian(end_ef)  # this is in world frame
    alpha = 2.
    cmd = alpha * (jac.T @ vel)  # using jacobian transpose
    cmd_arm = cmd * my_mask
    cmd_norm = cmd_arm / np.linalg.norm(cmd_arm, 1)
    robot.set_commands(cmd_norm)
    death_grips(grab_cond)


for step in range(total_steps):
    if (simu.schedule(simu.control_freq())):
        # Do something
        # task_controller(choose_target())
        # tree_draft()
        Behavior_Tree()
        box_translation = box.base_pose().translation()
        basket_translation = basket.base_pose().translation()
        if box_into_basket(
             box_translation, basket_translation, basket_z_angle):
            finish_counter += 1

        if (finish_counter >= 10):
            break

    if (simu.step_world()):
        break
