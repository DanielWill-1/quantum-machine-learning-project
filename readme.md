# Quantum Machine Learning for Image Classification

A research project exploring **Quantum Machine Learning (QML)** approaches for image classification using the MNIST dataset.

The project investigates whether quantum circuits can be integrated with classical neural networks to create **hybrid quantum-classical machine learning models**, and compares their performance against classical approaches.

> **Status:** 🚧 Active Research / Development

---

## Overview

This project explores the use of quantum computing techniques for image classification.

The current workflow is:

```text
MNIST Dataset
     │
     ▼
Image Preprocessing
     │
     ▼
Classical Feature Extraction
     │
     ▼
Quantum Circuit
     │
     ▼
Quantum Feature Representation
     │
     ▼
Classical Classifier
     │
     ▼
Prediction
```

The primary goal is to understand the practical advantages, limitations, and performance characteristics of **hybrid quantum-classical models**.

---

## Dataset

The project currently uses the **MNIST handwritten digit dataset**.

### Dataset Statistics

| Split      |  Images |
| ---------- | ------: |
| Training   |  60,000 |
| Testing    |  10,000 |
| Image Size | 28 × 28 |
| Classes    |      10 |
| Labels     |     0–9 |

The dataset is downloaded using **TensorFlow Datasets (TFDS)**.

The local dataset is intentionally excluded from Git using `.gitignore`.

```text
mnist/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
│
├── test/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
│
└── sample_10/
```

---

## Project Structure

```text
quantum-machine-learning-project/
│
├── .venv/                    # Local Python environment (ignored)
├── data/                     # Local data (ignored)
├── mnist/                    # MNIST dataset (ignored)
│
├── download.py               # Downloads and prepares MNIST
│
├── quantum_test.py           # Quantum circuit experiments
├── hybrid_classifier.py      # Hybrid quantum-classical classifier
│
├── a.py                      # Experimental model/training script
├── b.py                      # Experimental/auxiliary script
│
├── quantum_circuit_diagram.png
├── mnist_test_predictions.png
│
├── .gitignore
└── README.md
```

> Some scripts are currently experimental and may change as the research progresses.

---

## Installation

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd quantum-machine-learning-project
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` has not yet been generated:

```bash
pip install tensorflow tensorflow-datasets pillow
```

---

## Download MNIST

Run:

```bash
python download.py
```

This downloads the MNIST dataset and exports the images locally.

It also creates a small sample containing 10 test images for quick inspection.

---

## Running the Experiments

### Quantum Circuit Test

```bash
python quantum_test.py
```

This script is used to experiment with and validate the quantum circuit components.

### Hybrid Classifier

```bash
python hybrid_classifier.py
```

This runs the current hybrid quantum-classical classification experiment.

### Experimental Scripts

Additional experiments are currently contained in:

```bash
python a.py
```

and

```bash
python b.py
```

These scripts are under active development and may be reorganized as the project matures.

---

## Technologies

The project currently uses:

* **Python**
* **TensorFlow**
* **TensorFlow Datasets**
* **PyTorch**
* **Quantum Machine Learning framework(s)**
* **Pillow**
* **NumPy**

The exact quantum framework and model architecture may evolve during experimentation.

---

## Research Questions

The project is focused on questions such as:

1. Can quantum circuits effectively represent image features?
2. How does a hybrid quantum-classical model compare with a classical neural network?
3. How does the number of qubits affect classification performance?
4. How does circuit depth affect training and generalization?
5. What is the computational cost of quantum simulation compared with classical models?
6. Can quantum feature maps provide useful representations for image classification?
7. Under what conditions can QML provide an advantage over classical approaches?

---

## Evaluation

Models will be evaluated using standard classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Training time
* Inference time

Where applicable, experiments will be repeated across multiple random seeds to improve the reliability of comparisons.

---

## Classical vs Quantum Baseline

A major component of the project is maintaining a **classical baseline**.

The quantum model should not be evaluated in isolation. Its performance will be compared against an equivalent or appropriately matched classical model.

```text
                MNIST
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
     Classical Model    Hybrid QML Model
          │                │
          ▼                ▼
      Predictions       Predictions
          │                │
          └───────┬────────┘
                  ▼
             Comparison
```

This allows the experiment to determine whether the quantum component provides measurable benefits rather than simply producing a different implementation.

---

## Reproducibility

Experiments should record:

* Random seed
* Dataset split
* Number of qubits
* Circuit depth
* Learning rate
* Batch size
* Number of epochs
* Optimizer
* Model architecture
* Evaluation metrics

As the project develops, experiment configurations will be moved into dedicated configuration files to make experiments easier to reproduce.

---

## Current Status

### Completed

* [x] Python project environment
* [x] MNIST dataset download pipeline
* [x] Train/test dataset preparation
* [x] Sample image generation
* [x] Initial quantum circuit experiments
* [x] Initial hybrid classifier
* [x] Git repository setup

### In Progress

* [ ] Improve hybrid model architecture
* [ ] Establish strong classical baseline
* [ ] Run controlled experiments
* [ ] Compare classical and quantum models
* [ ] Perform hyperparameter experiments
* [ ] Analyze quantum circuit behavior
* [ ] Document experimental results

### Future Work

* [ ] Multiple quantum feature maps
* [ ] Different variational circuit architectures
* [ ] Noise simulation
* [ ] Hardware execution
* [ ] Scalability experiments
* [ ] Statistical significance analysis
* [ ] Final research report/paper

---

## Notes

This repository is an **experimental research project**. Model architectures, dependencies, and experiment configurations are expected to change as the research progresses.

The objective is not simply to achieve high MNIST accuracy, but to investigate the **practical role of quantum circuits within machine learning workflows**.

---

## License

This project is currently intended for research and educational purposes.
