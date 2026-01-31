# Deep Learning Foundations with TensorFlow and Keras

## Overview

This repository marks the **beginning of the Deep Learning course using TensorFlow and Keras**.  
The notebooks focus on **core mathematical and framework fundamentals** that are essential before building neural networks.

The emphasis is on:
- Understanding **gradient descent**
- Learning how **TensorFlow represents data using tensors**
- Building intuition for how optimization works in deep learning

---

## 1) Gradient Descent from Scratch  
**Notebook:** `GradientDescent.ipynb`

This notebook introduces **gradient descent**, the core optimization algorithm used to train neural networks.  
The implementation demonstrates how loss functions are minimized by iteratively updating parameters.

### Code snippets from the notebook

**Importing libraries**
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

**Gradient descent update**
```python
W = W - learning_rate * dW
b = b - learning_rate * db
```

### What this notebook covers
- Loss function behavior
- Gradients and parameter updates
- Effect of learning rate
- Visualization of convergence

---

## 2) TensorFlow Basics and Tensors  
**Notebook:** `TensorFlow.ipynb`

This notebook introduces **TensorFlow fundamentals**, focusing on how data is represented and manipulated using tensors.

### Code snippets from the notebook

**Creating tensors**
```python
import tensorflow as tf

scalar = tf.constant(7)
vector = tf.constant([10, 10])
matrix = tf.constant([[10, 7], [7, 10]])
```

**Tensor properties**
```python
tf.rank(tensor)
tf.shape(tensor)
```

## Requirements

```bash
pip install tensorflow numpy matplotlib
```

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
