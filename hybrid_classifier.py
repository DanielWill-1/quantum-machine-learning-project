import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

# 1. Define the Quantum Circuit (QNode)
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode classical data into quantum states (Angle Embedding)
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Trainable Quantum Gates (Strongly Entangling Layers)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure expectation values along Pauli-Z for both qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Declare tensor shapes for PennyLane parameters
weight_shapes = {"weights": (1, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# 2. Build the Hybrid Neural Network Model
class HybridQNN(nn.Module):
    def __init__(self):
        super(HybridQNN, self).__init__()
        # Classical Pre-processing: Redefine 4 features into 2 for the 2-qubit circuit
        self.fc1 = nn.Linear(4, 2)
        # Quantum Layer
        self.quantum = qlayer
        # Classical Post-processing: Map 2 quantum outputs to 2 class probabilities
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum(x)
        x = self.fc2(x)
        return x

# 3. Create Synthetic Dataset (Binary Classification, 4 Features)
X_train = torch.randn(100, 4)
y_train = torch.randint(0, 2, (100,))

# 4. Model Training Pipeline
model = HybridQNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting Hybrid Model Training...\n")
for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()  # Backpropagates through classical layers AND the quantum circuit
    optimizer.step()
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch [{epoch + 1:2d}/15] | Loss: {loss.item():.4f}")

# Sample Evaluation
with torch.no_grad():
    sample_input = torch.randn(1, 4)
    logits = model(sample_input)
    prediction = torch.argmax(logits, dim=1).item()
    print(f"\nSample Prediction Class: {prediction}")