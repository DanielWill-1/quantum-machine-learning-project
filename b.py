import pennylane as qml
import matplotlib.pyplot as plt

# 1. Define the exact same quantum circuit structure
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 2. Supply sample dummy inputs to populate gate structures
sample_inputs = [0.5, -0.8]
# Parameter shape: (1 layer, 2 qubits, 3 rotation angles per qubit)
sample_weights = qml.numpy.random.random((1, n_qubits, 3))

# 3. Render the circuit using PennyLane's Matplotlib drawer
fig, ax = qml.draw_mpl(quantum_circuit)(sample_inputs, sample_weights)

# Customize title and layout
plt.title("2-Qubit Hybrid QNN Layer Architecture", fontsize=12, pad=15)
plt.tight_layout()

# 4. Save diagram locally as an image file
output_filename = "quantum_circuit_diagram.png"
plt.savefig(output_filename, dpi=300)
print(f"Circuit diagram successfully saved as '{output_filename}'")

# Display inline plot window
plt.show()