# Linear Regression: From Theory to Implementation

A comprehensive implementation of Linear Regression with proper software architecture, covering mathematical foundations, optimization methods, regularization techniques, and practical applications.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Technical Explanation](#technical-explanation)
3. [Project Structure](#project-structure)
4. [Installation & Usage](#installation--usage)
5. [Experiments](#experiments)
6. [Key Concepts](#key-concepts)

---

## Mathematical Foundations

### 1. The Linear Regression Model

Linear regression finds the best straight line (or hyperplane in higher dimensions) that fits through your data points.

**Hypothesis Function:**

$$h(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

In matrix notation:

$$h(\mathbf{X}) = \mathbf{X}\boldsymbol{\theta}$$

Where:
- $\mathbf{X}$ is the feature matrix ($m$ samples $\times$ $n$ features)
- $\boldsymbol{\theta}$ is the parameter vector ($n+1$ parameters including bias)
- $h(\mathbf{X})$ is the predicted output

### 2. Cost Function (Mean Squared Error)

The cost function measures how well our model fits the data:

$$J(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$$

Where:
- $m$ = number of training examples
- $h(\mathbf{x}^{(i)})$ = predicted value for example $i$
- $y^{(i)}$ = actual value for example $i$
- The $\frac{1}{2}$ factor simplifies the derivative calculation

### 3. Optimization Methods

#### A. Normal Equation (Closed-form Solution)

Directly computes the optimal parameters:

$$\boldsymbol{\theta} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}$$

**Pros:**
- Exact solution (no iterations needed)
- No learning rate to tune

**Cons:**
- Computationally expensive: $\mathcal{O}(n^3)$
- Requires $\mathbf{X}^\top \mathbf{X}$ to be invertible
- Doesn't scale well with large datasets

#### B. Gradient Descent (Iterative Solution)

Iteratively updates parameters to minimize the cost function:

$$\theta_j := \theta_j - \alpha \frac{\partial J(\boldsymbol{\theta})}{\partial \theta_j}$$

The gradient is:

$$\frac{\partial J(\boldsymbol{\theta})}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

In matrix form:

$$\boldsymbol{\theta} := \boldsymbol{\theta} - \frac{\alpha}{m} \mathbf{X}^\top (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

Where:
- $\alpha$ = learning rate (step size)
- Updates all parameters simultaneously

**Variants:**
- **Batch GD**: Uses entire dataset (stable but slow)
- **Mini-batch GD**: Uses small batches (good balance)
- **Stochastic GD**: Uses one sample at a time (fast but noisy)

### 4. Regularization

Prevents overfitting by adding penalty terms to the cost function.

#### L2 Regularization (Ridge)

$$J(\boldsymbol{\theta}) = \frac{1}{2m} \left[ \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]$$

#### L1 Regularization (Lasso)

$$J(\boldsymbol{\theta}) = \frac{1}{2m} \left[ \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} |\theta_j| \right]$$

Where $\lambda$ controls the strength of regularization.

### 5. Evaluation Metrics

- **MSE (Mean Squared Error)**: Average squared difference
- **RMSE (Root MSE)**: Square root of MSE (same units as target)
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (0 to 1)
- **Adjusted R²**: R² adjusted for number of features

---

## Technical Explanation

### What Does "Best Fit" Mean?

"Finding the best straight line" technically means: **minimizing the sum of squared residuals** between predicted and actual values. This is called **Ordinary Least Squares (OLS)**.

### Why Squared Errors?

1. Penalizes large errors more heavily
2. Differentiable everywhere (enables gradient descent)
3. Has nice statistical properties (maximum likelihood under Gaussian noise)

### Feature Standardization

Before applying gradient descent, we standardize features:

$$x_{\text{standardized}} = \frac{x - \mu}{\sigma}$$

This ensures:
- All features contribute equally
- Faster convergence of gradient descent
- Better numerical stability

---

## Project Structure

```
linear_regression_project/
│
├── core/
│   └── model.py                    # Main LinearRegression class
│       ├── fit_normal_equation()   # Closed-form solution
│       ├── fit_gradient_descent()  # Iterative optimization
│       ├── predict()               # Make predictions
│       └── score()                 # Calculate R² score
│
├── utils/
│   └── metrics.py                  # Evaluation functions
│       ├── mean_squared_error()
│       ├── root_mean_squared_error()
│       ├── mean_absolute_error()
│       ├── r2_score()
│       └── adjusted_r2_score()
│
├── data/
│   └── loader.py                   # Data utilities
│       ├── generate_synthetic_data()
│       ├── load_diabetes_dataset()
│       ├── train_test_split()
│       └── standardize_features()
│
├── visualizations/
│   └── plots.py                    # Visualization tools
│       ├── plot_cost_history()
│       ├── plot_predictions_vs_actual()
│       ├── plot_residuals()
│       ├── plot_feature_importance()
│       └── plot_model_comparison()
│
├── experiments/
│   └── main_experiment.py          # Complete workflow
│       ├── Experiment 1: Synthetic data verification
│       ├── Experiment 2: Real-world diabetes dataset
│       └── Experiment 3: Hyperparameter tuning
│
├── outputs/                        # Generated results (auto-created)
│   └── *.png                       # Visualization plots
│
└── README.md                       # This file
```

### Design Principles

- **Separation of Concerns**: Each module has a single responsibility
- **Modularity**: Easy to test and extend individual components
- **Reusability**: Functions can be imported and used independently
- **Documentation**: Comprehensive docstrings and type hints

---

## Installation & Usage

### Prerequisites

```bash
pip install numpy scikit-learn matplotlib
```

### Running the Experiments

```bash
cd linear-regression
python experiments/main_experiment.py
```

### Basic Usage Example

```python
from core.model import LinearRegression
from data.loader import generate_synthetic_data, train_test_split, standardize_features
from utils.metrics import print_metrics

# Generate data
X, y = generate_synthetic_data(n_samples=1000, n_features=5, noise=0.1)

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_std, X_test_std, mu, sigma = standardize_features(X_train, X_test)

# Train model
model = LinearRegression()
model.fit_gradient_descent(X_train_std, y_train, learning_rate=0.01, n_iterations=1000)

# Evaluate
y_pred = model.predict(X_test_std)
print_metrics(y_test, y_pred, X_test_std.shape[1])
```

---

## Experiments

### Experiment 1: Synthetic Data Verification

**Goal**: Verify implementation correctness with known coefficients

**Setup**:
- Generate data with known true coefficients
- Add controlled noise
- Compare learned vs true coefficients

### Experiment 2: Real-World Application

**Goal**: Apply to medical dataset (diabetes progression)

**Dataset**: sklearn diabetes dataset (442 samples, 10 features)

**Analysis**:
- Feature importance ranking
- Residual analysis for model assumptions
- Comparison of different optimization methods

### Experiment 3: Hyperparameter Tuning

**Goal**: Find optimal learning rate and regularization strength

**Parameters Tested**:
- Learning rates: [0.001, 0.01, 0.1, 1.0]
- Regularization strengths: [0.0, 0.01, 0.1, 1.0]

**Results**: Visual comparison of convergence and performance

---

## Key Concepts

### Understanding Big-O Notation in Context

From our earlier discussion on algorithm complexity:

- $\mathcal{O}(n)$: Going through every element once (e.g., computing predictions)
- $\mathcal{O}(n \log n)$: Efficient sorting algorithms (not directly used here)
- $\mathcal{O}(n^2)$: Computing $\mathbf{X}^\top \mathbf{X}$ matrix multiplication
- $\mathcal{O}(n^3)$: Matrix inversion in Normal Equation

**Why it matters**: For large datasets ($n > 10{,}000$), the Normal Equation becomes impractical, making Gradient Descent the preferred choice.

### When to Use Which Method?

| Scenario | Recommended Method |
|----------|-------------------|
| Small dataset (< 1000 samples) | Normal Equation |
| Large dataset (> 10,000 samples) | Gradient Descent |
| Need exact solution | Normal Equation |
| Memory constraints | Stochastic GD |
| Online learning | Stochastic GD |
| Production systems | Mini-batch GD |

### Common Pitfalls

1. **Forgetting feature scaling**: GD converges slowly or not at all
2. **Learning rate too high**: Divergence (cost increases)
3. **Learning rate too low**: Very slow convergence
4. **Not checking residuals**: Missing model assumption violations
5. **Overfitting**: High $R^2$ on training, low on test set

---