# robot_model.py

import numpy as np

class RobotModel():
    def __init__(self, dt):
        self.dt = dt

    def predict_traj(self, curr_x, u):
        """Predict the trajectory using the kinematic model
        Args:
            curr_x (numpy.ndarray): Current state, [x, y, theta]
            u (numpy.ndarray): Control input, [v, w]
        Returns:
            next_state (numpy.ndarray): Next state prediction [x', y', theta']
        """
        x, y, theta = curr_x
        v, w = u
        dt = self.dt

        # Kinematic model: update position
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = w * dt

        next_state = np.array([x + dx, y + dy, theta + dtheta])
        return next_state
