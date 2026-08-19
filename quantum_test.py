import pennylane as qml
from pennylane import numpy as np

# 1. Initialize a CPU-based quantum simulator with 1 qubit
dev = qml.device("default.qubit", wires=1)

# 2. Define a variational quantum node (the quantum circuit)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# 3. Define a cost function to optimize parameters toward a target state
def cost(params):
    target = -0.5
    return (circuit(params) - target) ** 2

# 4. Initialize random parameters and a gradient descent optimizer
np.random.seed(42)
initial_params = np.array([0.5, 0.5], requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# 5. Run optimization steps
params = initial_params
print(f"Initial parameters: {params}, Initial cost: {cost(params):.4f}")

for step in range(20):
    params, prev_cost = opt.step_and_cost(cost, params)
    if (step + 1) % 5 == 0:
        print(f"Step {step + 1:2d} | Cost: {prev_cost:.4f}")

print(f"Optimized parameters: {params}")