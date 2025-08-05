````markdown
# Wrench-Based Dynamical Optimization Framework

**A Solid-Inspired Alternative to Gradient Descent in Linear Regression**  
by Yassine Mouadi — August 2025

---

## Overview

This project introduces a novel, physics-inspired optimizer for linear regression. It conceptualizes model parameters as components of a rigid body, whose adjustments are governed by both translational and rotational dynamics. This approach, termed the Wrench-Coupled Optimizer (WCO), serves as a conceptual and computational extension to traditional gradient descent. Unlike conventional gradient descent, which updates weights and bias independently, the WCO leverages physical intuition to update these parameters as a coupled system, incorporating principles of mass and inertia.

Learning is reimagined as motion, where:

- Force drives translational gradient descent.
- Torque facilitates coordinated rotational adjustments.
- The Wrench provides a unified representation of these combined dynamics.

---

## Contents

- `wrench_optimizer.py`: This script encapsulates the entire framework, including data preprocessing, implementations of both Gradient Descent (GD) and the Wrench-Coupled Optimizer (WCO), and comprehensive visualization tools.
- `report.pdf`: A detailed theoretical write-up and derivation of the Wrench-Coupled Optimizer.
- `data.csv`: The CSV file containing CO2 concentration data used for experimental validation.

---

## Requirements

This framework has been tested with Python 3.9 and later versions.  
To install the necessary libraries, please use the following command:

```bash
pip install numpy matplotlib pandas
````

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/wrench-optimizer.git
cd wrench-optimizer
```

Prepare the dataset: download the data.csv file within the repo.

Execute the optimizer script:

```bash
python wrench_optimizer.py
```

This script will perform the following operations:

* Load and preprocess the CO2 dataset.
* Normalize the input features.
* Execute the Gradient Descent optimization.
* Execute the Wrench-Coupled Optimizer.
* Compare the loss convergence of both methods.
* Generate and display prediction plots.
* Output the final denormalized parameters for both optimizers.

---

## How It Works

### Traditional Gradient Descent

The update rules for traditional gradient descent are:

$$
w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}
$$

This method is straightforward and effective, but it updates the weights (w) and bias (b) independently.

### Wrench-Coupled Optimizer

The update rules for the Wrench-Coupled Optimizer are:

$$
w \leftarrow w - \frac{1}{m} \cdot \frac{\partial J}{\partial w} - \frac{1}{I} \cdot \frac{\partial J}{\partial b}
$$

$$
b \leftarrow b - \frac{1}{m} \cdot \frac{\partial J}{\partial b} + \frac{1}{I} \cdot \sum \frac{\partial J}{\partial w}
$$

Where:

* $m$: Represents the 'mass' parameter, controlling the translational speed of the optimization.
* $I$: Represents the 'inertia' parameter, influencing the rotational twist in the parameter updates.
* $\frac{\partial J}{\partial w}, \frac{\partial J}{\partial b}$: Denote the gradients of the Mean Squared Error (MSE) loss function with respect to the weights and bias, respectively.

This formulation introduces a cross-coupling between $w$ and $b$ through the principles of solid body dynamics, analogous to a rigid object being simultaneously pulled and twisted. This mechanism facilitates a more nuanced movement across the loss surface, potentially leading to improved convergence characteristics.

---

## Results Summary

| Metric                 | Gradient Descent | Wrench-Coupled Optimizer |
| ---------------------- | ---------------- | ------------------------ |
| Final MSE (normalized) | 0.009477         | 0.009477                 |
| Final MSE (real scale) | 5.176450         | 5.176449                 |
| Final Weight (denorm.) | 0.0489           | 0.0489                   |
| Final Bias (denorm.)   | 328.7905         | 328.7905                 |

---

## Citation

If you utilize this work, please cite it as:
Yassine Mouadi, "A Wrench-Based Dynamical Optimization Framework: A Solid-Inspired Alternative to Gradient Descent in Linear Regression", August 2025.

---

## Author

Yassine Mouadi
LinkedIn: [linkedin.com/in/yassinemouadi](https://linkedin.com/in/yassinemouadi)
GitHub: [github.com/YassineMouadi](https://github.com/YassineMouadi)


---

## Contributions

Ideas, suggestions, bug reports, and forks are highly encouraged and welcome.

```
```
