import math
import numpy as np
from controller import Supervisor, Lidar, Motor
from PythonLinearNonlinearControl.controllers.nmpc import NMPC


# Constants
WHEEL_BASE_RADIUS = 0.2
WHEEL_RADIUS = 0.0985
MAX_WHEEL_VELOCITY = 3.0
ARRIVED_THRESH = 0.4

goal_point = np.array([6.0, 4.0, 0.0])

# UnicycleModel class compatible with NMPC
class UnicycleModel:
    def __init__(self, dt):
        self.dt = dt

    def f(self, x, u):
        theta = x[2]
        v = u[0]
        w = u[1]
        dx = np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            w
        ])
        return dx

    def predict_traj(self, x0, us):
        xs = [x0]
        x = x0.copy()
        for u in us:
            dx = self.f(x, u)
            x = x + self.dt * dx
            xs.append(x)
        return np.array(xs)

    def predict_adjoint_traj(self, xs, us, gs):
        lams = [np.zeros_like(xs[0])]
        for i in reversed(range(len(us))):
            x = xs[i]
            lam = lams[0]
            A = self.compute_A(x, us[i])
            Q = np.diag([10.0, 10.0, 0.1])
            q = Q @ (x - gs[i])
            lam = lam + self.dt * (q + A.T @ lam)
            lams.insert(0, lam)
        return np.array(lams[:-1])

    def compute_A(self, x, u):
        theta = x[2]
        v = u[0]
        A = np.array([
            [0.0, 0.0, -v * np.sin(theta)],
            [0.0, 0.0,  v * np.cos(theta)],
            [0.0, 0.0,  0.0]
        ])
        return A

    @staticmethod
    def gradient_hamiltonian_input(xs, lams, us, gs):
        grad = []
        R = np.diag([0.5, 0.01])
        for i in range(len(us)):
            u = us[i]
            x = xs[i]
            lam = lams[i]
            theta = x[2]
            grad_u = R @ u + np.array([
                lam[0] * np.cos(theta) + lam[1] * np.sin(theta),
                lam[2]
            ])
            grad.append(grad_u)
        return np.array(grad)

# Webots setup
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
ROBOT_NAME = "ROBOT1"

top_lidar = robot.getDevice("top_lidar")
top_lidar.enable(100)

left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

def get_position_and_heading():
    pos = np.array(robot.getSelf().getPosition())[:2]  # Only take x and y position
    rot = np.array(robot.getSelf().getOrientation()).reshape(3, 3)
    heading = math.atan2(rot[0, 1], rot[0, 0])  # Extract the heading from the rotation matrix
    return np.array([pos[0], pos[1], heading])  # Only return x, y, and heading


def set_velocities(v, w):
    diff_vel = w * WHEEL_BASE_RADIUS / WHEEL_RADIUS
    right_vel = (v + diff_vel) / WHEEL_RADIUS
    left_vel = (v - diff_vel) / WHEEL_RADIUS
    fastest = max(abs(right_vel), abs(left_vel))
    if fastest > MAX_WHEEL_VELOCITY:
        scale = MAX_WHEEL_VELOCITY / fastest
        left_vel *= scale
        right_vel *= scale
    left_motor.setVelocity(left_vel)
    right_motor.setVelocity(right_vel)

def distance_to_goal(state):
    return np.linalg.norm(state[:2] - goal_point[:2])

# NMPC Setup
Np = 10
dt = 0.1
Q = np.diag([10.0, 10.0, 0.1])  # 3x3 matrix
R = np.diag([0.5, 0.01])
Qf = Q

def quadratic_state_cost(x, xg, Q):
    # Ensure that x is a single 3D vector at each step
    assert len(x.shape) == 1 and x.shape == (3,), f"State should be a 3D vector, but got {x.shape}"  # Debugging check
    assert xg.shape == (3,), f"Goal state should be a 3D vector, but got {xg.shape}"  # Debugging check
    e = x - xg
    return 0.5 * np.dot(e.T, np.dot(Q, e))

def quadratic_input_cost(u, R):
    return 0.5 * u.T @ R @ u

def terminal_state_cost(x, xg, Qf):
    e = x - xg
    return 0.5 * e.T @ Qf @ e

class DummyConfig:
    def __init__(self):
        self.state_cost_fn = lambda x, xg: quadratic_state_cost(x, xg, Q)
        self.input_cost_fn = lambda u: quadratic_input_cost(u, R)
        self.terminal_state_cost_fn = lambda x, xg: terminal_state_cost(x, xg, Qf)
        self.PRED_LEN = Np
        self.INPUT_SIZE = 2
        self.DT = dt
        self.opt_config = {
            "NMPC": {
                "threshold": 1e-3,
                "max_iters": 15,
                "learning_rate": 0.4,
                "optimizer_mode": "conjugate"
            }
        }
        self.gradient_hamiltonian_input = UnicycleModel.gradient_hamiltonian_input

config = DummyConfig()
model = UnicycleModel(dt)
nmpc = NMPC(config, model)

def main():
    robot.step(500)
    goal_reached = False

    while robot.step(timestep) != -1:
        state = get_position_and_heading()

        # Ensure the state passed is a single 3D vector
        assert state.shape == (3,), f"State should be a 3D vector, but got {state.shape}"

        if distance_to_goal(state) < ARRIVED_THRESH:
            if not goal_reached:
                print(f"{ROBOT_NAME} reached goal.")
                goal_reached = True
            set_velocities(0, 0)
            continue

        # Set reference trajectory to goal_point, repeat the goal point Np+1 times
        ref_traj = np.tile(goal_point[:3], (Np + 1, 1))  # Ensure it's (Np+1, 3)
        print(f"Ref trajectory shape: {ref_traj.shape}")

        # Ensure only current state is passed to NMPC solver
        u = nmpc.obtain_sol(state, ref_traj)

        if u is not None:
            set_velocities(u[0], u[1])
        else:
            print("NMPC failed to solve.")
            set_velocities(0, 0)

if __name__ == "__main__":
    main()
