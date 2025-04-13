Deep Learning Assignment: Image Classification

Task 1

The repository contains two files for our implementations:

- scratch_trial_task1.py: MLP implemented from scratch using only NumPy
- pytorch_trial_task1.py: MLP implemented using PyTorch

Project Structure
-----------------
project/
├── dataset/
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
├── scratch_trial_task1.py
├── pytorch_trial_task2.py
├── README.txt

Requirements
------------
Make sure you have the following installed:

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib
- scikit-learn

Running the Scripts
-------------------

1. Run the Scratch Implementation

    python scratch_trial_task1.py

Make sure to run this first so that the results of the this implementation is saved to the directory.

2. Run the PyTorch Implementation

    python pytorch_trial.py

This script will both train the PyTorch implementation and plot comparative plots.

Notes
-----
- Make sure the dataset files are present in the dataset/ directory.
- The results might vary slightly across runs due to random initialization, which happened on our own case.


# Deep Learning Assignment: Image Classification

This repository contains implementations for CMPE597 Assignment 1. Both tasks are implemented in two versions: one from scratch using only NumPy, and one using PyTorch.

## Project Structure

```
project/
├── dataset/
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
├── scratch_trial_task1.py      # Task 1 – Scratch implementation (NumPy)
├── pytorch_trial_task1.py      # Task 1 – PyTorch implementation
├── task2_scratch.py            # Task 2 – Scratch implementation (NumPy + GloVe)
├── task2_pytorch.py            # Task 2 – PyTorch implementation
├── glove.6B.50d.txt            # GloVe embeddings (downloaded from: https://nlp.stanford.edu/projects/glove/)
├── README.md
```

## Requirements

Ensure the following packages are installed:

- Python 3.8+
- NumPy
- PyTorch
- scikit-learn
- Matplotlib

You can install the required libraries via:

```bash
pip install numpy torch scikit-learn matplotlib
```

---

## Task 1

### 1.1 Scratch Implementation



```bash
python scratch_trial_task1.py
```


### 1.2 PyTorch Implementation


```bash
python pytorch_trial_task1.py
```

---

## Task 2

### 2.1 Scratch Implementation

```bash
python task2_scratch.py
```


Make sure you have the GloVe file (`glove.6B.50d.txt`) in the root directory.

---

### 2.2 PyTorch Implementation

```bash
python task2_pytorch.py
```


---

## Notes

- Ensure `dataset/` folder contains the required `.npy` files.
- Place `glove.6B.50d.txt` in the same directory as the code.
- Task 2 models use fixed `random.seed=42` for reproducibility.


