# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:07:35 2025

@author: Pol
"""

import numpy as np

def infer_abundance_MLE(y, T, epsilon, seuil_correction, alpha_prior, beta_prior, theta_values, theta_change_times=None):
    """
    Effectue l'inférence des paramètres à partir des données simulées, en gérant les valeurs manquantes,
    et en prenant en compte un \(\theta\) constant ou variable.

    Arguments :
    - y : Données observées (peut contenir des np.nan).
    - T : Nombre total d'observations.
    - epsilon : Petite valeur pour éviter les divisions par zéro.
    - seuil_correction : Seuil pour absence de changement.
    - alpha_prior : Paramètre alpha du prior Gamma.
    - beta_prior : Paramètre beta du prior Gamma.
    - theta_values : Liste des valeurs possibles pour \(\theta\).
    - theta_change_times : Liste des temps de changement associés à \(\theta\) (en indices, optionnel).

    Retourne :
    - tau_est : Temps de changement estimé.
    - alpha_post_N0, beta_post_N0 : Paramètres postérieurs de N0.
    - alpha_post_N1, beta_post_N1 : Paramètres postérieurs de N1.
    - nan_percentage : Pourcentage de valeurs `np.nan` dans les données.
    """
    # Vérifier si \(\theta\) est constant
    if not theta_change_times or len(theta_values) == 1:
        theta_change_times = []  # Pas de changement
        theta_values = [theta_values[0]]  # Utiliser la première valeur de \(\theta\)

    # Étendre les temps de changement pour inclure les bornes de début et de fin
    extended_times = [0] + theta_change_times + [T]
    extended_theta = [theta_values[i] for i in range(len(theta_values))]

    def get_theta_at(t):
        """
        Retourne la valeur de \(\theta\) pour un instant \(t\).
        """
        for i in range(len(extended_times) - 1):
            if extended_times[i] <= t < extended_times[i + 1]:
                return extended_theta[i]
        return extended_theta[-1]  # Valeur par défaut pour la dernière période

    def posterior_params(y, theta, alpha_prior, beta_prior):
        """
        Calcule les paramètres a posteriori pour une distribution Gamma.
        """
        valid_data = y[~np.isnan(y)]  # Exclure les np.nan
        alpha_post = alpha_prior + np.sum(valid_data)
        beta_post = beta_prior + theta * len(valid_data)
        return alpha_post, beta_post

    def log_likelihood_components(tau, y):
        """
        Calcule les trois composantes de la log-vraisemblance pour un tau donné,
        en gérant les np.nan et en prenant en compte des changements indépendants pour \(\tau\) et \(\theta\).

        Retourne :
        - sum_y_ln_y : Composante \(\sum_t y_t \ln(y_t)\)
        - sum_y_ln_theta : Composante \(\sum_t -y_t \ln(\theta_t)\)
        - sum_y_over_theta : Composante \(\sum_t -y_t / \theta_t\)
        - ll_total : Log-vraisemblance totale
        """
        tau = int(np.clip(tau, 1, T - 1).item())
        all_change_times = sorted(set([tau] + theta_change_times + [0, T]))
        
        sum_y_ln_y = 0
        sum_y_ln_theta = 0
        sum_y_over_theta = 0

        for i in range(len(all_change_times) - 1):
            start, end = all_change_times[i], all_change_times[i + 1]
            segment_data = y[start:end]
            valid_data = segment_data[~np.isnan(segment_data)]

            if len(valid_data) > 0:
                theta_segment = get_theta_at(start)
                mean_segment = max(np.mean(valid_data), epsilon)

                sum_y_ln_y += np.sum(valid_data * np.log(mean_segment))
                sum_y_ln_theta += np.sum(-valid_data * np.log(theta_segment))
                sum_y_over_theta += np.sum(-valid_data / theta_segment)
        ll_total = sum_y_ln_y + sum_y_ln_theta + sum_y_over_theta
        return sum_y_ln_y, sum_y_ln_theta, sum_y_over_theta, ll_total

    # Calcul du pourcentage de valeurs manquantes
    nan_count = np.sum(np.isnan(y))
    nan_percentage = (nan_count / len(y)) * 100

    # Recherche par grille pour tau estimé
    tau_range = range(1, T - 1)
    components = [log_likelihood_components(tau, y) for tau in tau_range]
    sum_y_ln_y_vals, sum_y_ln_theta_vals, sum_y_over_theta_vals, log_likelihoods = zip(*components)
    tau_est = tau_range[np.argmax(log_likelihoods)]

    
    # Fusionner les temps de changement pour \(\theta\) et \(\tau\)
    all_change_times = sorted(set([0, tau_est] + theta_change_times + [T]))
    
    # Calcul des paramètres postérieurs pour chaque segment
    alpha_post_N0, beta_post_N0 = 0, 0
    alpha_post_N1, beta_post_N1 = 0, 0
    
    for i in range(len(all_change_times) - 1):
        start = all_change_times[i]
        end = all_change_times[i + 1]
    
        # Déterminer si ce segment est avant ou après tau
        is_before_tau = start < tau_est
    
        # Obtenir la valeur de \(\theta\) pour ce segment
        theta_segment = get_theta_at(start)
    
        # Extraire les données pour ce segment
        segment_data = y[start:end]
        valid_data = segment_data[~np.isnan(segment_data)]
    
        if len(valid_data) > 0:
            # Calculer les contributions au postérieur en fonction de la position par rapport à tau
            alpha_segment = alpha_prior + np.sum(valid_data)
            beta_segment = beta_prior + theta_segment * len(valid_data)
    
            if is_before_tau:
                alpha_post_N0 += alpha_segment
                beta_post_N0 += beta_segment
            else:
                alpha_post_N1 += alpha_segment
                beta_post_N1 += beta_segment
    
    # Si aucune donnée n'existe pour un segment, garder les valeurs par défaut
    if alpha_post_N0 == 0:
        beta_post_N0 = beta_prior
    if alpha_post_N1 == 0:
        beta_post_N1 = beta_prior


    # MAP pour N0 et N1
    N0_est = max((alpha_post_N0 - 1) / beta_post_N0, 0) if alpha_post_N0 > 1 else 0
    N1_est = max((alpha_post_N1 - 1) / beta_post_N1, 0) if alpha_post_N1 > 1 else 0

    # Vérification de la similarité entre les tailles estimées
    if abs(N1_est - N0_est) * 2 / (N0_est + N1_est + epsilon) < seuil_correction:
        # Correction appliquée
        N_common = (N0_est + N1_est) / 2
        N0_est = N1_est = N_common
        tau_est = T  # Absence de changement

    return tau_est, alpha_post_N0, beta_post_N0, alpha_post_N1, beta_post_N1, nan_percentage
