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
