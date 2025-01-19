import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Simulate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 for intercept

# Cost function
def compute_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Gradient descent steps
def gradient_step(theta, X, y, learning_rate):
    m = len(y)
    gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
    return theta - learning_rate * gradients

def stochastic_step(theta, X, y, learning_rate):
    m = len(y)
    rand_idx = np.random.randint(m)
    xi = X[rand_idx:rand_idx+1]
    yi = y[rand_idx:rand_idx+1]
    gradients = xi.T.dot(xi.dot(theta) - yi)
    return theta - learning_rate * gradients

def mini_batch_step(theta, X, y, learning_rate, batch_size):
    m = len(y)
    batch_indices = np.random.choice(m, batch_size, replace=False)
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    gradients = (1 / batch_size) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
    return theta - learning_rate * gradients

# Initialize parameters
theta_bgd = np.random.randn(2, 1)
theta_sgd = np.random.randn(2, 1)
theta_mbgd = np.random.randn(2, 1)
learning_rate = 0.1
iterations = 50
batch_size = 20

# Store history
theta_history_bgd = [theta_bgd.copy()]
theta_history_sgd = [theta_sgd.copy()]
theta_history_mbgd = [theta_mbgd.copy()]
cost_history_bgd = [compute_cost(theta_bgd, X_b, y)]
cost_history_sgd = [compute_cost(theta_sgd, X_b, y)]
cost_history_mbgd = [compute_cost(theta_mbgd, X_b, y)]

# Perform gradient descent
for _ in range(iterations):
    theta_bgd = gradient_step(theta_bgd, X_b, y, learning_rate)
    theta_sgd = stochastic_step(theta_sgd, X_b, y, learning_rate)
    theta_mbgd = mini_batch_step(theta_mbgd, X_b, y, learning_rate, batch_size)

    theta_history_bgd.append(theta_bgd.copy())
    theta_history_sgd.append(theta_sgd.copy())
    theta_history_mbgd.append(theta_mbgd.copy())

    cost_history_bgd.append(compute_cost(theta_bgd, X_b, y))
    cost_history_sgd.append(compute_cost(theta_sgd, X_b, y))
    cost_history_mbgd.append(compute_cost(theta_mbgd, X_b, y))

# Prepare contour data
theta0_vals = np.linspace(-10, 15, 150)  # Expanded range
theta1_vals = np.linspace(-10, 15, 150)
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
Z = np.array([
    compute_cost(np.array([[t0], [t1]]), X_b, y)
    for t0, t1 in zip(np.ravel(T0), np.ravel(T1))
]).reshape(T0.shape)

# Animation
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

def init():
    for ax in axs.flat:
        ax.clear()

def update(i):
    # Contour plot
    axs[0, 0].clear()
    axs[0, 0].contour(T0, T1, Z, levels=30, cmap="viridis")  # Circular contour lines
    axs[0, 0].plot(
        [theta[0] for theta in theta_history_bgd[:i+1]],
        [theta[1] for theta in theta_history_bgd[:i+1]],
        color="red", label="Batch GD Path"
    )
    axs[0, 0].plot(
        [theta[0] for theta in theta_history_sgd[:i+1]],
        [theta[1] for theta in theta_history_sgd[:i+1]],
        color="blue", label="Stochastic GD Path"
    )
    axs[0, 0].plot(
        [theta[0] for theta in theta_history_mbgd[:i+1]],
        [theta[1] for theta in theta_history_mbgd[:i+1]],
        color="green", label="Mini-Batch GD Path"
    )
    axs[0, 0].scatter(*theta_history_bgd[i].ravel(), color="red")
    axs[0, 0].scatter(*theta_history_sgd[i].ravel(), color="blue")
    axs[0, 0].scatter(*theta_history_mbgd[i].ravel(), color="green")
    axs[0, 0].set_title("Contour Plot")
    axs[0, 0].legend()

    # Regression line plot
    axs[0, 1].clear()
    axs[0, 1].scatter(X, y, color="gray", alpha=0.7)
    axs[0, 1].plot(X, X_b.dot(theta_history_bgd[i]), color="red", label="Batch GD")
    axs[0, 1].plot(X, X_b.dot(theta_history_sgd[i]), color="blue", label="Stochastic GD")
    axs[0, 1].plot(X, X_b.dot(theta_history_mbgd[i]), color="green", label="Mini-Batch GD")
    axs[0, 1].set_xlim(0, 2)
    axs[0, 1].set_ylim(0, 15)
    axs[0, 1].set_title("Best Fit Line Updates")
    axs[0, 1].legend()

    # Cost function plot
    axs[1, 0].clear()
    axs[1, 0].plot(range(i+1), cost_history_bgd[:i+1], color="red", label="Batch GD")
    axs[1, 0].plot(range(i+1), cost_history_sgd[:i+1], color="blue", label="Stochastic GD")
    axs[1, 0].plot(range(i+1), cost_history_mbgd[:i+1], color="green", label="Mini-Batch GD")
    axs[1, 0].set_title("Cost Function vs Iterations")
    axs[1, 0].legend()

    # Parameter trajectory plot
    axs[1, 1].clear()
    axs[1, 1].plot(
        [theta[0] for theta in theta_history_bgd[:i+1]],
        [theta[1] for theta in theta_history_bgd[:i+1]],
        color="red", label="Batch GD"
    )
    axs[1, 1].plot(
        [theta[0] for theta in theta_history_sgd[:i+1]],
        [theta[1] for theta in theta_history_sgd[:i+1]],
        color="blue", label="Stochastic GD"
    )
    axs[1, 1].plot(
        [theta[0] for theta in theta_history_mbgd[:i+1]],
        [theta[1] for theta in theta_history_mbgd[:i+1]],
        color="green", label="Mini-Batch GD"
    )
    axs[1, 1].set_title("Parameter Trajectory")
    axs[1, 1].legend()

ani = FuncAnimation(fig, update, frames=iterations, init_func=init, interval=100)
ani.save("gradient_descent_complete.gif", writer=PillowWriter(fps=10))

plt.show()
