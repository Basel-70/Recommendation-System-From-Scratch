# Recommendation System From Scratch

A simple recommendation system built from scratch using **Python** and **NumPy**.

This project implements the core ideas behind recommendation systems without using machine learning frameworks such as Scikit-learn, TensorFlow, or PyTorch.

---

## Project Goal

The goal of this project is to understand how machine learning models learn internally by manually implementing:

* Matrix Factorization
* Dot Product Prediction
* Loss Function
* Gradient Descent
* Learning Rate
* Epoch Training
* Weight Updates

---

## How It Works

The model represents each user and each anime as a vector of latent factors.

The predicted rating is calculated using the dot product:

```python
prediction = np.dot(user_weights, anime_weights)
```

Then the model calculates the loss:

```python
loss = (actual - prediction) ** 2
```

After that, gradients are calculated manually and the weights are updated using Gradient Descent.

---

## Technologies Used

* Python
* NumPy

---

## Why I Built This

I built this project to understand how recommendation systems learn from data instead of only using ready-made machine learning libraries.

This project helped me understand what happens behind functions like:

```python
loss.backward()
optimizer.step()
```

---

## Future Improvements

* Train on a real dataset such as MovieLens
* Add validation and test sets
* Plot the loss curve
* Rebuild the same model using PyTorch
* Build a simple API for recommendations

---

## Author

Created by **Basel Alzahrani**
