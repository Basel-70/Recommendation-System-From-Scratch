"""
Recommendation System From Scratch

Created by: Basel Alzahrani

This project is a simple implementation of a recommendation system
using Matrix Factorization, Gradient Descent, and Loss Functions
without relying on machine learning libraries such as Scikit-learn,
TensorFlow, or PyTorch.

The goal of this project is to understand how recommendation systems
learn by manually implementing prediction, loss calculation,
gradient computation, and weight updates.
"""

import numpy as np

users = ["Basel","Taif","Mohannad"]
anime = ["Naruto", "Mushoku Tensei", "Attack on Titan"]

ratings = np.array([[5,4, np.nan],
                    [np.nan,5,4],
                    [2,np.nan,5]])

np.random.seed(42)
user_weights = np.random.rand(3,2)
anime_weights = np.random.rand(3,2)

# Hyperparameters
lr = 0.001
epochs = 2500

for epoch in range(epochs):
   total_loss = 0

   for i in range(len(users)):
      for j in range(len(anime)):

         actual = ratings[i][j]

         # Skip unknown ratings during training
         if np.isnan(actual):
            continue

         # Predict the rating using the dot product
         prediction = np.dot(user_weights[i], anime_weights[j])

         loss = (actual - prediction) ** 2
       
         # Calculate prediction error
         error = prediction - actual

         # Compute gradients
         gradient_user = 2 * error * anime_weights[j]
         gradient_anime = 2 * error * user_weights[i]

         # Update weights using Gradient Descent
         user_weights[i] = user_weights[i] - lr * gradient_user
         anime_weights[j] = anime_weights[j] - lr * gradient_anime

         total_loss += loss

   if epoch % 100 == 0:
      print("Epoch:", epoch, "Total Loss:", total_loss)


print("\nPredictions for missing ratings:")

for i in range(len(users)):
   for j in range(len(anime)):

      if np.isnan(ratings[i][j]):
         prediction = np.dot(user_weights[i], anime_weights[j])
         prediction = np.clip(prediction, 1, 5)

         print(users[i], "might rate", anime[j], "=", round(prediction, 2))
