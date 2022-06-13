import numpy as np
import RobotDART as rd

def box_into_basket(box_translation, basket_translation, basket_angle):
    basket_xy_corners = np.array([basket_translation[0] + 0.14, basket_translation[0] + 0.14, basket_translation[0] - 0.14, basket_translation[0] - 0.14,
                                  basket_translation[1] - 0.08, basket_translation[1] + 0.08, basket_translation[1] + 0.08, basket_translation[1] - 0.08], dtype=np.float64).reshape(2, 4)

    rotation_matrix = np.array([np.cos(basket_angle), np.sin(basket_angle), -np.sin(basket_angle), np.cos(basket_angle)], dtype=np.float64).reshape(2, 2)

    basket_center = np.array([basket_translation[0], basket_translation[1]], dtype=np.float64).reshape(2, 1)
    rotated_basket_xy_corners = np.matmul(rotation_matrix, (basket_xy_corners - basket_center)) + basket_center

    d1 = (rotated_basket_xy_corners[0][1] - rotated_basket_xy_corners[0][0]) * (box_translation[1] - rotated_basket_xy_corners[1][0]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][0]) * (rotated_basket_xy_corners[1][1] - rotated_basket_xy_corners[1][0])
    d2 = (rotated_basket_xy_corners[0][2] - rotated_basket_xy_corners[0][1]) * (box_translation[1] - rotated_basket_xy_corners[1][1]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][1]) * (rotated_basket_xy_corners[1][2] - rotated_basket_xy_corners[1][1])
    d3 = (rotated_basket_xy_corners[0][3] - rotated_basket_xy_corners[0][2]) * (box_translation[1] - rotated_basket_xy_corners[1][2]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][2]) * (rotated_basket_xy_corners[1][3] - rotated_basket_xy_corners[1][2])
    d4 = (rotated_basket_xy_corners[0][0] - rotated_basket_xy_corners[0][3]) * (box_translation[1] - rotated_basket_xy_corners[1][3]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][3]) * (rotated_basket_xy_corners[1][0] - rotated_basket_xy_corners[1][3])

    if ((d1 > 0.0) and (d2 > 0.0) and (d3 > 0.0) and (d4 > 0.0) and (box_translation[2] <= 0.04)):
        return True
    else:
        return False

def create_grid(box_step_x=0.5, box_step_y=0.5, basket_step_x=1., basket_step_y=1.):
    basket_positions = []
    basket_x_min = -2.
    basket_x_max = 2.
    basket_y_min = -2.
    basket_y_max = 2.

    basket_nx_steps = int(np.floor((basket_x_max-basket_x_min) / basket_step_x))
    basket_ny_steps = int(np.floor((basket_y_max-basket_y_min) / basket_step_y))

    for x in range(basket_nx_steps+1):
        for y in range(basket_ny_steps+1):
            basket_x = basket_x_min + x * basket_step_x
            basket_y = basket_y_min + y * basket_step_y
            if (np.linalg.norm([basket_x, basket_y]) < 2.):
                continue
            basket_positions.append((basket_x, basket_y))

    box_positions = []
    box_x_min = -1.
    box_x_max = 1.
    box_y_min = -1.
    box_y_max = 1.

    box_nx_steps = int(np.floor((box_x_max-box_x_min) / box_step_x))
    box_ny_steps = int(np.floor((box_y_max-box_y_min) / box_step_y))

    for x in range(box_nx_steps+1):
        for y in range(box_ny_steps+1):
            box_x = box_x_min + x * box_step_x
            box_y = box_y_min + y * box_step_y
            if (np.linalg.norm([box_x, box_y]) < 1.):
                continue
            box_positions.append((box_x, box_y))

    return (basket_positions, box_positions)