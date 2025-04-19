import numpy as np

def state_cost_fn(pred_xs, g_xs):
    return np.sum((pred_xs - g_xs) ** 2, axis=1)

def terminal_state_cost_fn(pred_x, g_x):
    return np.sum((pred_x - g_x) ** 2)

def input_cost_fn(inputs):
    return np.sum(inputs ** 2, axis=1)
