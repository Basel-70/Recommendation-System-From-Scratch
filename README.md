# Recommendation System From Scratch

A recommendation system built completely from scratch using **Python** and **NumPy**, without relying on machine learning frameworks such as **Scikit-learn**, **TensorFlow**, or **PyTorch**.

The purpose of this project is to understand the mathematics and learning process behind recommendation systems by implementing every training step manually.

---

# Project Goal

This project was created to gain a deeper understanding of how recommendation systems learn from data instead of relying on high-level machine learning libraries.

The model is implemented completely from scratch, including:

* Matrix Factorization
* Dot Product Prediction
* Loss Function
* Gradient Calculation
* Gradient Descent
* Learning Rate
* Epoch Training
* Manual Weight Updates

---

# How It Works

Each user and each anime is represented as a latent feature vector.

The predicted rating is calculated using the dot product between the user vector and the anime vector:

```python
prediction = np.dot(user_weights[i], anime_weights[j])
```

The model then calculates the prediction error using the Mean Squared Error (MSE) loss:

```python
loss = (actual - prediction) ** 2
```

Using Gradient Descent, the gradients are computed manually, and both the user and anime vectors are updated during every training epoch.

---

# Technologies Used

* Python
* NumPy

---

# Project Limitations

This project uses a very small educational dataset.

Because of the limited amount of data, the model can easily **overfit** by memorizing the known ratings.

This behavior is expected and is acceptable for the educational purpose of this project.

The objective is to understand the complete learning process rather than to build a production-ready recommendation system.

---

# Why I Built This

I wanted to understand what happens behind functions such as:

```python
loss.backward()
optimizer.step()
```

Instead of treating them as black boxes.

Building the entire learning pipeline manually helped me understand how machine learning models update their parameters during training.

---

# Future Improvements

* Train the model using the MovieLens dataset.
* Add Training, Validation, and Test splits.
* Implement L2 Regularization.
* Plot the training loss curve.
* Calculate RMSE for evaluation.
* Rebuild the project using PyTorch.
* Develop a REST API for real-world recommendation systems.

---

# Author

**Basel Alzahrani**
