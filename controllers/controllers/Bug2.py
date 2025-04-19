"""bug1 controller."""

from math import sqrt, acos
from controller import Supervisor, Lidar, Motor, Node, Field
import numpy as np
from enum import Enum

WHEEL_BASE_RADIUS = 0.2  # taken from Webots PROTO
WHEEL_RADIUS = 0.0985  # taken from Webots PROTO
TARGET_FORWARD_VELOCITY = 0.3  # m/s, tweak this to your liking.
MAX_WHEEL_VELOCITY = 3  # TIGAo in webots is set to 10.15
LIDAR_FOV = 6.28  # radians
OBJECT_HIT_DIST = 1  # when do we consider a hit-point? in m.
OBJECT_FOLLOW_DIST = 1.4  # How closely do we follow the obstacle? in m.
ARRIVED_THRESH = 0.4

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
goal_node = robot.getFromDef("TrafficCone")
GOAL_POINT = goal_node.getField("translation").getSFVec3f()
# LIDAR
top_lidar = Lidar("top_lidar")
# The user guide recommends to use robot.getDevice("top_lidar"), but pylint has
# a hard time finding the type of the return value. This also style works fine
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
    angle = acos(
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
        translation {} {} {}
        children [
            Shape {{
            appearance PBRAppearance {{
                baseColor 0.8 0 0
            }}
            geometry Sphere {{
                radius 0.2
            }}
            }}
        ]
    }}
    """.format(
            robot_translation[0], robot_translation[1], robot_translation[2]
        ),
    )


class State(Enum):
    Travelling = 1
    FollowWall = 2
    MetGoal = 3


RobotState = State.Travelling


def main():
    ranges = top_lidar.getRangeImage()
    current_location = get_my_location()
    global RobotState, initial_location, leave_point, wall_met_location

    match RobotState:
        case State.Travelling:
            # The robot is travelling, i.e. there are no objects in our sensor range
            x_track_error = distance_to_line(leave_point, GOAL_POINT)
            heading_error = heading_difference(leave_point, GOAL_POINT)
            pure_pursuit(x_track_error, heading_error)
            print("Heading towards goal")

            # Change state if there is an obstacle ahead
            if any(r < OBJECT_HIT_DIST for r in ranges):
                wall_met_location = current_location
                place_marker(wall_met_location)
                RobotState = State.FollowWall

        case State.FollowWall:
            # Calculate distance along m-line by length of vector projection
            wall_met_to_current = np.array(current_location) - np.array(
                wall_met_location
            )
            leave_to_goal = np.array(GOAL_POINT) - np.array(leave_point)
            mline_dist_traversed = np.dot(leave_to_goal, wall_met_to_current) / (
                np.linalg.norm(leave_to_goal)
            )

            # Check if the line has been met again
            if (
                mline_dist_traversed > ARRIVED_THRESH
                and abs(distance_to_line(leave_point, GOAL_POINT)) < ARRIVED_THRESH
            ):
                leave_point = get_my_location()
                place_marker(leave_point)
                RobotState = State.Travelling

            x_track_error = distance_to_line(leave_point, GOAL_POINT)
            heading_error = heading_difference(leave_point, GOAL_POINT)
            pure_pursuit(x_track_error, heading_error)
            print("Following Wall")

            angle = angle_to_object(ranges)
            dist = min(ranges)
            # We will try and keep the set distance. Turning to right, so 90 degrees is added to the reference.
            pure_pursuit(OBJECT_FOLLOW_DIST - dist, angle - 1.578)

        case State.MetGoal:
            print("Program Terminating")
            set_velocities(0, 0)
            robot.simulationSetMode(0)
            exit()

    # Change state if goal is met
    if distance_to(GOAL_POINT) < ARRIVED_THRESH:
        RobotState = State.MetGoal


if __name__ == "__main__":
    # Wait a moment to get the first lidar scan
    robot.step(100)
    # initialize leave-point to initial location
    initial_location = get_my_location()
    leave_point = initial_location
    wall_met_point = 0

    while robot.step(timestep) != -1:
        main()
