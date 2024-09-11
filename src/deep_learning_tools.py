import numpy as np
import plotly.express as px
import pandas as pd


def gradient_descent(X, y, alpha=0.01, tolerance=1e-6, max_iterations=1000):
  # Données d'exemple
  X = np.array([1, 2, 3, 4, 5])
  y = np.array([2, 3, 4, 5, 6])

  # Initialisation des paramètres
  beta_0 = 0
  beta_1 = 0

  # Descente de gradient
  for i in range(max_iterations):
      # Calcul des prédictions
      y_pred = beta_0 + beta_1 * X

      # Calcul des dérivées partielles
      d_beta_0 = -2 * np.sum(y - y_pred) / len(y)
      d_beta_1 = -2 * np.sum((y - y_pred) * X) / len(y)

      # Mise à jour des paramètres
      beta_0_new = beta_0 - alpha * d_beta_0
      beta_1_new = beta_1 - alpha * d_beta_1

      # Vérification de la convergence
      if np.abs(beta_0_new - beta_0) < tolerance and np.abs(beta_1_new - beta_1) < tolerance:
          print(f"Convergence atteinte après {i+1} itérations.")
          break

      # Mise à jour des paramètres
      beta_0 = beta_0_new
      beta_1 = beta_1_new

  print(f"Intercept (beta_0): {beta_0}")
  print(f"Pente (beta_1): {beta_1}")
