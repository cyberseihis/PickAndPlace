import numpy as np
import dartpy as dp
import RobotDART as rd

from utils import create_grid, box_into_basket

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
robot.set_commands([0.1, 0., 0.],
                   ['rootJoint_rot_z', 'rootJoint_pos_x', 'rootJoint_pos_y'])

# Create position grid for the box/basket
basket_positions, box_positions = create_grid()

# Create box
box_size = [0.04, 0.04, 0.04]
# Random cube position
box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0],
            box_positions[box_pt][1], box_size[2] / 2.0]
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


def rpose():
    rbp = robot.base_pose()
    rbt = rbp.translation()
    return rbt


def bpose():
    return box.base_pose().translation()


def delta_rot():
    vec = bpose() - rpose()
    radt = np.arctan2(vec[0], vec[1])
    return radt


def angdif(x, y):
    sd = np.sin(x-y)
    cd = np.cos(x-y)
    return np.arctan2(sd, cd)


def angle_err():
    dero = delta_rot()
    rbr = robot.base_pose().rotation()
    ebrz = dp.math.matrixToEulerXYZ(rbr)[2]
    difz = angdif(dero, ebrz)
    return difz


def stop_rot():
    robot.set_commands([0., 0., 0.],
                       ['rootJoint_rot_z',
                        'rootJoint_pos_x',
                        'rootJoint_pos_y'])


for step in range(total_steps):
    if (simu.schedule(simu.control_freq())):
        # Do something
        box_translation = box.base_pose().translation()
        basket_translation = basket.base_pose().translation()
        an_e = angle_err()
        if np.abs(an_e) < 0.01:
            stop_rot()
        else:
            print(f"{an_e:.4f}")
        if box_into_basket(
             box_translation, basket_translation, basket_z_angle):
            finish_counter += 1

        if (finish_counter >= 10):
            break

    if (simu.step_world()):
        break
