import numpy as np

def gradient_descent(X, y, alpha=0.01, tolerance=1e-6, max_iterations=1000):
    """
    Implémente l'algorithme de descente de gradient pour la régression linéaire.

    Cette fonction utilise l'algorithme de descente de gradient pour ajuster les paramètres
    d'un modèle de régression linéaire simple (y = beta_0 + beta_1 * X).

    Paramètres :
    X (array-like) : Les valeurs des caractéristiques (features).
    y (array-like) : Les valeurs cibles (targets).
    alpha (float) : Le taux d'apprentissage (learning rate). Par défaut, 0.01.
    tolerance (float) : La tolérance pour déterminer la convergence. Par défaut, 1e-6.
    max_iterations (int) : Le nombre maximum d'itérations. Par défaut, 1000.

    Retourne :
    None
    """
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
