def line_search(gradient, current_input, cost_function, init_alpha=0.1, beta=0.8, c=1e-4, max_iters=20):
    alpha = init_alpha
    cost_0 = cost_function(current_input)
    grad_dot = (gradient * gradient).sum()

    for _ in range(max_iters):
        new_input = current_input - alpha * gradient
        cost_new = cost_function(new_input)

        if cost_new <= cost_0 - c * alpha * grad_dot:
            break
        alpha *= beta

    return alpha
