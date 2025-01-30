# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:46:42 2025

@author: Pol
"""

import numpy as np

import numpy as np

def infer_abundance_MLE_complex_theta(y, T, epsilon, seuil_correction, alpha_prior, beta_prior, 
                                      theta_values, theta_change_times=None, theta_transition_type="constant"):
    """
    Effectue l'inférence des paramètres à partir des données simulées, en gérant les valeurs manquantes,
    et en prenant en compte différentes variations de \(\theta\) (constant, linéaire, exponentiel, etc.).

    Arguments :
    - y : Données observées (peut contenir des np.nan).
    - T : Nombre total d'observations.
    - epsilon : Petite valeur pour éviter les divisions par zéro.
    - seuil_correction : Seuil pour absence de changement.
    - alpha_prior : Paramètre alpha du prior Gamma.
    - beta_prior : Paramètre beta du prior Gamma.
    - theta_values : Liste des valeurs possibles pour \(\theta\), **avec une valeur de plus que `theta_change_times`**.
    - theta_change_times : Liste des temps de changement associés à \(\theta\) (en indices, optionnel).
    - theta_transition_type : Type de transition entre les valeurs de theta. 
                              Options : "constant", "linear", "exponential", "logarithmic", "sigmoid".

    Retourne :
    - tau_est : Temps de changement estimé.
    - alpha_post_N0, beta_post_N0 : Paramètres postérieurs de N0.
    - alpha_post_N1, beta_post_N1 : Paramètres postérieurs de N1.
    - nan_percentage : Pourcentage de valeurs `np.nan` dans les données.
    """

    # Vérifier si `theta_change_times` est None (cas de theta constant)
    if theta_change_times is None:
        theta_change_times = []  # Pas de changement
        theta_values = [theta_values[0]]  # Utiliser la première valeur de \(\theta\)

    # Vérification du bon nombre de valeurs de theta
    if len(theta_values) != len(theta_change_times) + 1:
        raise ValueError(f"Le nombre de valeurs dans `theta_values` ({len(theta_values)}) "
                         f"doit être égal au nombre de `theta_change_times` + 1 ({len(theta_change_times) + 1}).")

    # Étendre les temps de changement pour inclure les bornes de début et de fin
    extended_times = [0] + theta_change_times + [T]
    extended_theta = theta_values.copy()

    def get_theta_at(t):
        """
        Retourne la valeur de \(\theta\) pour un instant \(t\) en fonction du type de transition sélectionné.
        """
        for i in range(len(extended_times) - 1):
            t_start, t_end = extended_times[i], extended_times[i + 1]
            theta_start, theta_end = extended_theta[i], extended_theta[i + 1]

            if t_start <= t < t_end:
                if theta_transition_type == "constant":
                    return theta_start
                elif theta_transition_type == "linear":
                    return theta_start + (theta_end - theta_start) * (t - t_start) / (t_end - t_start)
                elif theta_transition_type == "exponential":
                    return theta_start * np.exp(np.log(theta_end / theta_start) * (t - t_start) / (t_end - t_start))
                elif theta_transition_type == "logarithmic":
                    return theta_start + (theta_end - theta_start) * np.log(1 + 9 * (t - t_start) / (t_end - t_start)) / np.log(10)
                elif theta_transition_type == "sigmoid":
                    return theta_start + (theta_end - theta_start) / (1 + np.exp(-10 * ((t - t_start) / (t_end - t_start) - 0.5)))
                else:
                    raise ValueError(f"Type de transition {theta_transition_type} non reconnu.")

        return extended_theta[-1]  # Valeur par défaut pour la dernière période

    # Calcul du pourcentage de valeurs manquantes
    nan_count = np.sum(np.isnan(y))
    nan_percentage = (nan_count / len(y)) * 100

    # Calcul de la log-vraisemblance et estimation de tau
    tau_range = range(1, T - 1)
    log_likelihoods = []
    
    for tau in tau_range:
        theta_vals_tau = np.array([get_theta_at(t) for t in range(tau)])
        theta_vals_remaining = np.array([get_theta_at(t) for t in range(tau, T)])
        log_likelihood = -np.sum(y[:tau] * np.log(theta_vals_tau) - theta_vals_tau) - np.sum(y[tau:] * np.log(theta_vals_remaining) - theta_vals_remaining)
        log_likelihoods.append(log_likelihood)

    tau_est = tau_range[np.argmax(log_likelihoods)]

    # Calcul des postérieurs pour N0 et N1
    alpha_post_N0 = alpha_prior + np.sum(y[:tau_est])
    beta_post_N0 = beta_prior + np.sum([get_theta_at(t) for t in range(tau_est)])
    
    alpha_post_N1 = alpha_prior + np.sum(y[tau_est:])
    beta_post_N1 = beta_prior + np.sum([get_theta_at(t) for t in range(tau_est, T)])

    # MAP pour N0 et N1
    N0_est = max((alpha_post_N0 - 1) / beta_post_N0, 0) if alpha_post_N0 > 1 else 0
    N1_est = max((alpha_post_N1 - 1) / beta_post_N1, 0) if alpha_post_N1 > 1 else 0

    # Vérification de la similarité entre les tailles estimées
    if abs(N1_est - N0_est) * 2 / (N0_est + N1_est + epsilon) < seuil_correction:
        N_common = (N0_est + N1_est) / 2
        N0_est = N1_est = N_common
        tau_est = T  # Absence de changement

    return tau_est, alpha_post_N0, beta_post_N0, alpha_post_N1, beta_post_N1, nan_percentage
