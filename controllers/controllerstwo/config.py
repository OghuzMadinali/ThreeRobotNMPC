# config.py

class Config:
    PRED_LEN = 20  # Prediction horizon
    INPUT_SIZE = 2  # [linear_velocity, angular_velocity]
    threshold = 0.01  # Convergence threshold
    max_iters = 100  # Max iterations for NMPC optimization
    learning_rate = 0.01  # Gradient descent learning rate
