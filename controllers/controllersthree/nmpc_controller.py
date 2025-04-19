from controller import Controller
from cost_functions import state_cost_fn, terminal_state_cost_fn, input_cost_fn
from utils import line_search


class NMPC(Controller):
    def __init__(self, config, model):
        """ Nonlinear Model Predictive Control using gradient-based optimization
        """
        super(NMPC, self).__init__(config, model)

        # Model
        self.model = model

        # Get cost functions
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

        # Controller parameters
        self.threshold = config.opt_config["NMPC"]["threshold"]
        self.max_iters = config.opt_config["NMPC"]["max_iters"]
        self.learning_rate = config.opt_config["NMPC"]["learning_rate"]
        self.optimizer_mode = config.opt_config["NMPC"]["optimizer_mode"]

        # General parameters
        self.pred_len = config.PRED_LEN
        self.input_size = config.INPUT_SIZE
        self.dt = config.DT

        # Initialize previous solution
        self.prev_sol = np.zeros((self.pred_len, self.input_size))

    def obtain_sol(self, curr_x, g_xs):
        """ Calculate the optimal inputs using NMPC.
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarray): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        # Initialize solution with the previous solution (converted to NumPy array)
        sol = np.array(self.prev_sol.copy())  # Ensure it's a NumPy array
        count = 0
        
        # Initialize conjugate gradient method variables
        conjugate_d = None
        conjugate_prev_d = None
        conjugate_s = None
        conjugate_beta = None

        while True:
            # Predict the trajectory using the current solution
            pred_xs = self.model.predict_traj(curr_x, sol)
            # Predict adjoint trajectory
            pred_lams = self.model.predict_adjoint_traj(pred_xs, sol, g_xs)

            # Compute the gradient of the Hamiltonian with respect to inputs
            F_hat = self.config.gradient_hamiltonian_input(pred_xs, pred_lams, sol, g_xs)

            # Check if the norm of the gradient is below the threshold
            if np.linalg.norm(F_hat) < self.threshold:
                break

            # Check if maximum iterations are exceeded
            if count > self.max_iters:
                logger.debug(f"Breaking at max iteration: F = {np.linalg.norm(F_hat)}")
                break

            # Conjugate gradient update if selected as optimizer mode
            if self.optimizer_mode == "conjugate":
                conjugate_d = F_hat.flatten()

                if conjugate_prev_d is None:  # First iteration
                    conjugate_s = conjugate_d
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)
                else:
                    prev_d = np.dot(conjugate_prev_d, conjugate_prev_d)
                    d = np.dot(conjugate_d, conjugate_d - conjugate_prev_d)
                    conjugate_beta = (d + 1e-6) / (prev_d + 1e-6)

                    conjugate_s = conjugate_d + conjugate_beta * conjugate_s
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)

            # Define the evaluation function for line search
            def compute_eval_val(u):
                pred_xs = self.model.predict_traj(curr_x, u)
                state_cost = np.sum(self.config.state_cost_fn(pred_xs[1:-1], g_xs[1:-1]))
                input_cost = np.sum(self.config.input_cost_fn(u))
                terminal_cost = np.sum(self.config.terminal_state_cost_fn(pred_xs[-1], g_xs[-1]))
                return state_cost + input_cost + terminal_cost

            # Perform line search to determine the optimal step size (alpha)
            alpha = line_search(F_hat, sol, compute_eval_val, init_alpha=self.learning_rate)

            # Update solution using gradient descent
            sol -= alpha * F_hat
            count += 1

        # Update prev_sol for next optimization step
        self.prev_sol = np.concatenate((sol[1:], np.zeros((1, self.input_size))), axis=0)

        # Return the optimal input (first element of the solution trajectory)
        return sol[0]
