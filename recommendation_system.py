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

User = ["Basel","Taif","Mohannad"]
Anime = ["Naruto", "Mushoku Tensei", "Attack on Titan"]

ratings = np.array([[5,4, np.nan],
                    [np.nan,5,4],
                    [2,np.nan,5]])

User_Weights = np.random.rand(3,2)
Anime_Weights = np.random.rand(3,2)

# Hyperparameters
lr = 0.001
epochs = 2500

for epoch in range(epochs):
   total_loss = 0

   for i in range(len(User)):
      for j in range(len(Anime)):

         actual = ratings[i][j]

        # Skip unknown ratings during training
         if np.isnan(actual):
            continue

        # Predict the rating using the dot product
         prediction = np.dot(User_Weights[i], Anime_Weights[j])

         loss = (actual - prediction) ** 2
       
        # Calculate prediction error
         error = prediction - actual

        # Compute gradients
         gradient_User = 2 * error * Anime_Weights[j]
         gradient_Anime = 2 * error * User_Weights[i]

        # Update weights using Gradient Descent
         User_Weights[i] = User_Weights[i] - lr * gradient_User
         Anime_Weights[j] = Anime_Weights[j] - lr * gradient_Anime

         total_loss += loss

   if epoch % 100 == 0:
      print("Epoch:", epoch, "Total Loss:", total_loss)


print("\nPredictions for missing ratings:")


for i in range(len(User)):
   for j in range(len(Anime)):

      if np.isnan(ratings[i][j]):
         prediction = np.dot(User_Weights[i], Anime_Weights[j])
         prediction = np.clip(prediction, 1, 5)

         print(User[i], "might rate", Anime[j], "=", round(prediction, 2))
