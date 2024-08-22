# Newton's Method

def derivative(f):
    def f_prime(x):
        epsilon = 0.001
        delta_y = f(x+epsilon) - f(x)
        return delta_y / epsilon
    return f_prime

# 1 for maximization, -1 for minimization
def optimize(x0, f, mode=-1):
    epsilon = 0.01
    f_prime = derivative(f)
    f_double_prime = derivative(f_prime)
    x_prev = x0
    x = x_prev + mode * f_prime(x_prev) / f_double_prime(x_prev)
    while abs(x - x_prev) > epsilon:
        x_prev = x
        x = x_prev + mode * f_prime(x_prev) / f_double_prime(x_prev)
    return x, f(x)