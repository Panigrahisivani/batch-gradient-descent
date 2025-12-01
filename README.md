# Batch Gradient Descent From Scratch

This repository contains a complete implementation of **Batch Gradient Descent** for linear regression using Python and NumPy.  
Both **single-variate** and **multi-variate** gradient descent are implemented, along with evaluation metrics and visualizations.

## Overview

Batch Gradient Descent updates parameters using the entire dataset at each iteration. It is stable, deterministic, and widely used for linear regression.

### Hypothesis  
```
ŷ = m*x + b
```

### Cost Function (Mean Squared Error)  
```
J = (1/n) * Σ (y - ŷ)²
```

### Parameter Update Rules  
```
m = m - α * (-2/n) * Σ[x * (y - ŷ)]
b = b - α * (-2/n) * Σ[(y - ŷ)]
```

---

## Repository Structure

```
notebooks/
    single_variate_gd.ipynb
    multi_variate_gd.ipynb

plots/
    loss_curve.png
    actual_vs_predicted.png

Data/
    House_Price_Prediction.csv
```

---

## Loss Curve Example

![Loss Curve](plots/loss_curve.png)

---

## Features

- Single-variate batch gradient descent  
- Multi-variate batch gradient descent  
- Loss curve visualization  
- Actual vs predicted comparison  
- Residual analysis  
- Performance comparison with scikit-learn  

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Running the Project

```
pip install -r requirements.txt
jupyter notebook
```
