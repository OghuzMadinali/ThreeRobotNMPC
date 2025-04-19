import math
from controller import Supervisor, Lidar, Motor
import numpy as np
import sys


WHEEL_BASE_RADIUS = 0.2
WHEEL_RADIUS = 0.0985
TARGET_FORWARD_VELOCITY = 0.3
MAX_WHEEL_VELOCITY = 3
LIDAR_FOV = 6.28
ARRIVED_THRESH = 0.4
SAFE_DIST = 0.7

print(sys.executable)

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# SET THESE UNIQUELY IN EACH FILE
ROBOT_NAME = "ROBOT3"
GOAL_POINT = np.array([2.0, 4.0, 0.0])
OTHER_ROBOT_NAMES = ["ROBOT1", "ROBOT2"]  # Higher priority robots first!

top_lidar = robot.getDevice("top_lidar")
top_lidar.enable(100)

left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

goal_reached_flags = {name: False for name in OTHER_ROBOT_NAMES}


def set_velocities(linear_vel, angular_vel):
    diff_vel = angular_vel * WHEEL_BASE_RADIUS / WHEEL_RADIUS
    right_vel = (linear_vel + diff_vel) / WHEEL_RADIUS
    left_vel = (linear_vel - diff_vel) / WHEEL_RADIUS
    fastest = max(abs(right_vel), abs(left_vel))
    if fastest > MAX_WHEEL_VELOCITY:
        scale = MAX_WHEEL_VELOCITY / fastest
        left_vel *= scale
        right_vel *= scale
    left_motor.setVelocity(left_vel)
    right_motor.setVelocity(right_vel)


def get_position(node=None):
    if node is None:
        return np.array(robot.getSelf().getPosition())
    return np.array(node.getPosition())


def angle_to_object(ranges):
    index_min = min(range(len(ranges)), key=ranges.__getitem__)
    angle = LIDAR_FOV * 0.5 - LIDAR_FOV * float(index_min) / float(len(ranges) - 1)
    return angle


def heading_difference(start, end):
    rot = np.array(robot.getSelf().getOrientation()).reshape(3, 3)
    heading = np.dot(rot, np.array([1, 0, 0]))
    line = end - start
    angle = math.acos(np.dot(heading, line) / (np.linalg.norm(line) * np.linalg.norm(heading)))
    cross = np.cross(line, heading)
    if cross[2] > 0.0:
        angle = -angle
    return angle


def distance_to(goal):
    return np.linalg.norm(get_position() - goal)


def distance_to_line(start, end):
    A = np.array(start)
    B = np.array(end)
    P = get_position()
    n = np.cross([0, 0, 1], (B - A))
    return np.dot((P - A), n) / np.linalg.norm(n)


def attractive_gradient(curr, goal):
    diff = curr - goal
    dist = np.linalg.norm(diff)
    zeta = 1
    d_star = 4
    return zeta * diff if dist <= d_star else (d_star * zeta * diff) / dist


def repulsive_gradient(curr_pos, lidar_ranges, other_robot_positions):
    grad = np.zeros(3)
    Q_star = 7
    eta = 40

    # LIDAR obstacle
    d_obs = min(lidar_ranges)
    if not math.isinf(d_obs) and d_obs < Q_star:
        angle = angle_to_object(lidar_ranges)
        obs_pos = curr_pos + d_obs * np.array([math.cos(angle), math.sin(angle), 0])
        diff = curr_pos - obs_pos
        grad += eta * ((1 / Q_star) - (1 / d_obs)) * (1 / d_obs ** 2) * (diff / d_obs)

    # Repulsion from other robots
    for pos in other_robot_positions:
        diff = curr_pos - pos
        dist = np.linalg.norm(diff)
        if dist < Q_star:
            grad += eta * ((1 / Q_star) - (1 / dist)) * (1 / dist ** 2) * (diff / dist)

    return grad


def get_other_robot_positions_and_status():
    positions = []
    for name in OTHER_ROBOT_NAMES:
        node = robot.getFromDef(name)
        if node:
            pos = get_position(node)
            dist_to_goal = np.linalg.norm(pos - GOAL_POINT)
            goal_reached_flags[name] = dist_to_goal < ARRIVED_THRESH
            positions.append((name, pos))
    return positions


def should_yield(curr_pos, others):
    for name, pos in others:
        dist = np.linalg.norm(curr_pos - pos)
        if dist < SAFE_DIST and not goal_reached_flags[name]:
            return True
    return False


def main():
    robot.step(500)
    goal_reached = False

    while robot.step(timestep) != -1:
        curr_pos = get_position()
        other_data = get_other_robot_positions_and_status()

        if distance_to(GOAL_POINT) < ARRIVED_THRESH:
            if not goal_reached:
                print(f"{ROBOT_NAME} reached goal.")
                goal_reached = True
            set_velocities(0, 0)
            continue

        if should_yield(curr_pos, other_data):
            set_velocities(0, 0)
            continue

        lidar_ranges = top_lidar.getRangeImage()
        other_positions = [pos for _, pos in other_data]

        grad = attractive_gradient(curr_pos, GOAL_POINT) + repulsive_gradient(curr_pos, lidar_ranges, other_positions)
        step_size = 0.1
        target_pos = curr_pos - step_size * grad
        x_error = distance_to_line(curr_pos, target_pos)
        heading_error = heading_difference(curr_pos, target_pos)
        set_velocities(TARGET_FORWARD_VELOCITY, -0.05 * x_error + 0.1 * heading_error)


if __name__ == "__main__":
    main()
