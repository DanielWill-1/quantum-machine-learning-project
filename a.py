import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import pennylane as qml

# ---------------------------------------------------------
# 1. Quantum Layer Setup (2-Qubit Variational Circuit)
# ---------------------------------------------------------
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode features into quantum rotation angles
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Trainable entangling gates
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure expectation values on Pauli-Z operator
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (1, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ---------------------------------------------------------
# 2. Hybrid Model Definition (CNN Feature Extractor + QNN)
# ---------------------------------------------------------
class HybridMNISTModel(nn.Module):
    def __init__(self):
        super(HybridMNISTModel, self).__init__()
        # Classical CNN to reduce 28x28 image to 2 extracted features
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        self.fc_reduce = nn.Linear(8 * 4 * 4, 2)
        
        # Quantum Neural Network Layer
        self.quantum = qlayer
        
        # Output layer mapping quantum measurements to 2 class logits
        self.classifier = nn.Linear(2, 2)

    def forward(self, x):
        # Image Feature Extraction
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc_reduce(x))  # Output bounded to [-pi, pi] for quantum embedding
        
        # Quantum Processing
        x = self.quantum(x)
        
        # Final Classification
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 3. Load & Filter MNIST Data (0 vs 1 only)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Filter for images labeled 0 or 1
indices = [i for i, (_, label) in enumerate(full_dataset) if label in [0, 1]]
subset_indices = indices[:500]  # Select 500 samples for fast local CPU execution
mnist_subset = Subset(full_dataset, subset_indices)

train_loader = DataLoader(mnist_subset, batch_size=16, shuffle=True)

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
model = HybridMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("Starting Hybrid Quantum-Classical MNIST Training (Digits 0 and 1)...")
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    accuracy = (correct / total) * 100
    print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")

print("Training finished!")

# ---------------------------------------------------------
# 5. Model Evaluation & Testing
# ---------------------------------------------------------
import matplotlib.pyplot as plt

# Load the separate MNIST test set
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Filter for test images labeled 0 or 1
test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in [0, 1]]
test_subset = Subset(test_dataset, test_indices[:100])  # Evaluate on 100 test samples
test_loader = DataLoader(test_subset, batch_size=1, shuffle=True)

# Set model to evaluation mode
model.eval()

correct = 0
total = 0

print("\n--- Evaluating Hybrid Model on Test Data ---")
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = (correct / total) * 100
print(f"Test Accuracy on unseen digits (0 & 1): {test_accuracy:.2f}%\n")

# ---------------------------------------------------------
# 6. Visualize Sample Visual Predictions
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
model.eval()

with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):
        if i >= 5:
            break
        output = model(image)
        _, pred = torch.max(output, 1)
        
        # Display image and prediction
        ax = axes[i]
        ax.imshow(image.squeeze().numpy(), cmap='gray')
        ax.set_title(f"True: {label.item()}\nPred: {pred.item()}")
        ax.axis('off')

plt.tight_layout()
plt.savefig("mnist_test_predictions.png", dpi=300)
print("Saved visual predictions to 'mnist_test_predictions.png'")
plt.show()