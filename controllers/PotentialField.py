"""bug1 controller."""

###################################################################
############# Remember to load the world "Bug.wbt" #############
###################################################################

import math
from controller import Supervisor, Lidar, Motor, Node, Field
import numpy as np

GOAL_POINT = [10.0, 4, 0.0]

WHEEL_BASE_RADIUS = 0.2  # taken from Webots PROTO
WHEEL_RADIUS = 0.0985  # taken from Webots PROTO
TARGET_FORWARD_VELOCITY = 0.3  # m/s, tweak this to your liking.
MAX_WHEEL_VELOCITY = 3  # TIGAo in webots is set to 10.15
LIDAR_FOV = 6.28  # radians
OBJECT_HIT_DIST = 1  # when do we consider a hit-point? in m.
OBJECT_FOLLOW_DIST = 1.4  # How closely do we follow the obstacle? in m.
ARRIVED_THRESH = 0.4

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())  # In milliseconds

# LIDAR
top_lidar = Lidar("top_lidar")
# The user guide recommends to use robot.getDevice("top_lidar"), but pylint has
# a hard time finding the type of the return value. This style also works fine
# for me.
if top_lidar is None:
    print("Could not find a lidar.")
    exit()
top_lidar.enable(100)

# MOTORS
# Find motors and set them up to velocity control
left_motor = Motor("wheel_left_joint")
# Same as with the lidar above: robot.getDevice("wheel_left_joint")
right_motor = Motor("wheel_right_joint")

if left_motor is None or right_motor is None:
    print("Could not find motors.")
    exit()
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
# left_motor.setAcceleration(-1)
# right_motor.setAcceleration(-1)
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


def set_velocities(linear_vel, angular_vel):
    """Set linear and angular velocities of the robot.

    Arguments:
    linear_velocity  -- Forward velocity in m/s.
    angular_velocity -- Rotational velocity rad/s. Positive direction is
                        counter-clockwise.
    """
    diff_vel = angular_vel * WHEEL_BASE_RADIUS / WHEEL_RADIUS
    right_vel = (linear_vel + diff_vel) / WHEEL_RADIUS
    left_vel = (linear_vel - diff_vel) / WHEEL_RADIUS
    fastest_wheel = abs(max(right_vel, left_vel))
    if fastest_wheel > MAX_WHEEL_VELOCITY:
        left_vel = left_vel / fastest_wheel * MAX_WHEEL_VELOCITY
        right_vel = right_vel / fastest_wheel * MAX_WHEEL_VELOCITY
    left_motor.setVelocity(left_vel)
    right_motor.setVelocity(right_vel)


def pure_pursuit(x_track_error, angle_error):
    """Follow a line.

    Arguments:
    x_track_error -- Cross-track error. The distance error perpendicular to the
                     reference line. If the robot is located to the left of the
                     line,it gives a positive error; negative to the right.
    angle_error   -- Angle error is the difference between the angle of the line
                     and the heading of the robot. If the reference line points
                     towards North and the robot to East, the error is positive;
                     negative to West.
    """
    ang_vel = -0.05 * x_track_error + 0.1 * angle_error
    set_velocities(TARGET_FORWARD_VELOCITY, ang_vel)


def get_my_location():
    """Gets the robots translation in the world."""
    return robot.getSelf().getPosition()


def angle_to_object(ranges):
    """Find the angle that points to the closest lidar point.

    We will assume that the lidar is mounted symmetrically, so that the center
    lidar point is at the front of the robot. We will return this as 0 rads.
    """
    index_min = min(range(len(ranges)), key=ranges.__getitem__)
    angle = LIDAR_FOV * 0.5 - LIDAR_FOV * float(index_min) / float(len(ranges) - 1)
    return angle


def heading_difference(start_point, end_point):
    """Get the angle difference between the robot heading and a line

    The line is defined by a start and an end point. The angle between the
    heading and the line is given by the dot product. To get orientation
    we need to look at the direction of the cross product along the z-axis.
    """
    rot_robot = np.array(robot.getSelf().getOrientation()).reshape(3, 3)
    robot_heading = np.dot(rot_robot, np.array([1, 0, 0]))
    line = np.array(end_point) - np.array(start_point)
    angle = math.acos(
        np.dot(robot_heading, line)
        / np.linalg.norm(line)
        / np.linalg.norm(robot_heading)
    )
    cross = np.cross(line, robot_heading)
    if cross[2] > 0.0:
        angle = -angle

    return angle


def distance_to(goal_location):
    """Get the current distance to a goal"""
    my_location = get_my_location()
    dist = np.linalg.norm(np.array(goal_location) - np.array(my_location))
    return dist


def distance_to_line(start_point, end_point):
    """Compute robot's perpendicular distance from the robot to a line.

    The line is defined by a start- and an end-point.

    The robot is assumed to move only in the x/y plane. So a normal vector (n)
    is computed between the line (B-A) and the z-unit vector. The projection of
    the current location (P) onto the normal vector gives the distance.
    """
    A = np.array(start_point)
    B = np.array(end_point)
    P = np.array(get_my_location())
    n = np.cross([0, 0, 1], (B - A))
    distance = np.dot((P - A), n) / np.linalg.norm(n)
    return distance


def place_marker(robot_translation):
    """Place a marker at the robot's location."""
    root = robot.getRoot()
    world_children = root.getField("children")
    world_children.importMFNodeFromString(
        -1,
        """
    Transform {{
        translation {} 0.1 {}
        children [
            Shape {{
            appearance PBRAppearance {{
                baseColor 0.8 0 0
            }}
            geometry Sphere {{
                radius 0.1
            }}
            }}
        ]
    }}
    """.format(
            robot_translation[0], robot_translation[2]
        ),
    )


def repulsive_gradient(curr_pos, ranges):
    # Based on equation 4.5 in the book

    nabla_U_rep = np.zeros(3)

    # Determine distance to closest obstacle
    dist_to_obstacle = min(ranges)
    # Check if 'inf' was returned
    if math.isinf(dist_to_obstacle):
        return nabla_U_rep

    # Obstacle position w.r.t. global frame
    obstacle_pos = curr_pos + dist_to_obstacle * np.array(
        [math.cos(angle_to_object(ranges)), math.sin(angle_to_object(ranges)), 0]
    )

    diff_to_obstacle = curr_pos - obstacle_pos

    # Calculate gradient for repelling function
    Q_star = 10  # Distance between transition to zero repel
    eta = 15  # Gain on the repulsive gradient
    if dist_to_obstacle <= Q_star:
        # The gradient vector for distance function is estimated using Eqn. 4.7
        nabla_dist = diff_to_obstacle / dist_to_obstacle
        nabla_U_rep = (
            eta
            * ((1 / Q_star) - (1 / dist_to_obstacle))
            * (1 / pow(dist_to_obstacle, 2))
            * nabla_dist
        )

    return nabla_U_rep


def attractive_gradient(curr_pos, goal_pos):
    # Based on equation 4.3 in the book
    nabla_U_att = np.zeros(3)

    diff_to_goal = curr_pos - goal_pos

    dist_to_goal = distance_to(GOAL_POINT)

    # Calculate gradient for attractive function, using Eqn. 4.3
    zeta = 1  # magnitude of gradient vector that points away from goal
    d_star = 4
    if dist_to_goal <= d_star:
        # Quadratic function
        nabla_U_att = zeta * (diff_to_goal)
    else:
        # Conical function
        nabla_U_att = (d_star * zeta * (diff_to_goal)) / dist_to_goal

    # print(nabla_U_att)
    return nabla_U_att


def potential_gradient(dist_to_goal, dist_to_obstacle, diff_to_goal, diff_to_obstacle):
    # Based on equation 4.5 and 4.3 in the book
    nabla_U_att, nabla_U_rep = np.zeros(2)

    # Calculate gradient for attractive function, using Eqn. 4.3
    zeta = 4  # magnitude of gradient vector that points away from goal
    d_star = 4
    if dist_to_goal <= d_star:
        # Quadratic function
        nabla_U_att = zeta * (diff_to_goal)
    else:
        # Conical function
        nabla_U_att = (d_star * zeta * (diff_to_goal)) / dist_to_goal

    # Calculate gradient for repelling function
    Q_star = 5  # Distance between transition to zero repel
    eta = 15  # Gain on the repulsive gradient
    if dist_to_obstacle <= Q_star:
        # The gradient vector for distance function is estimated using Eqn. 4.7
        nabla_dist = diff_to_obstacle / dist_to_obstacle
        nabla_U_rep = (
            eta
            * ((1 / Q_star) - (1 / dist_to_obstacle))
            * (1 / pow(dist_to_obstacle, 2))
            * nabla_dist
        )
    # print(nabla_U_att + nabla_U_rep)
    return nabla_U_att + nabla_U_rep


def main():
    # Wait a moment to get the first lidar scan
    robot.step(500)

    # Define goal position
    goal_pos = np.array(GOAL_POINT)

    while robot.step(timestep) != -1:

        if distance_to(GOAL_POINT) < OBJECT_HIT_DIST:
            print("Goal has been reached")
            set_velocities(0, 0)
            return

        ranges = top_lidar.getRangeImage()

        # Current position
        curr_pos = np.array(get_my_location())  # x,y,z

        # Calculate gradient vector
        nabla_U = attractive_gradient(curr_pos, goal_pos) + repulsive_gradient(
            curr_pos, ranges
        )
        print(nabla_U)

        # Take a step
        alpha = 0.1  # A small step size to ensure smooth motion
        x_track_error = distance_to_line(curr_pos, curr_pos - alpha * nabla_U)
        heading_error = heading_difference(curr_pos, curr_pos - alpha * nabla_U)
        pure_pursuit(x_track_error, heading_error)


if __name__ == "__main__":
    main()
