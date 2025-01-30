# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:36:07 2025

@author: Pol
"""

import numpy as np
import itertools

def infer_abundance_MLE_theta_tau(y, T, epsilon=1e-6, alpha_prior=6.25, beta_prior=0.01, max_theta_segments=3):
    """
    Inférence conjointe de la taille de population N et des paramètres d’échantillonnage θ
    en estimant simultanément :
        - tau : Changement de population
        - tau_theta : Changements de θ
        - k_theta : Nombre de segments pour θ
        - N0, N1 : Tailles de population
        - θ_t : Paramètres d’échantillonnage
        
    Utilise le Maximum de Vraisemblance avec sélection du modèle optimal selon l'AIC.

    Arguments :
    - y : Données observées.
    - T : Nombre total d'observations.
    - epsilon : Valeur pour éviter les divisions par zéro.
    - alpha_prior, beta_prior : Paramètres du prior Gamma pour \( N \).
    - max_theta_segments : Nombre maximum de segments pour θ.

    Retourne :
    - Meilleurs paramètres estimés : (tau, tau_theta, N0, N1, theta_values, log_likelihood_best, AIC_best).
    """

    def compute_log_likelihood(tau, tau_theta, theta_values):
        """
        Calcule la log-vraisemblance pour un choix donné de (τ, τ_θ, θ).
        """
        ll_total = 0
        for t in range(T):
            theta_t = theta_values[np.searchsorted(tau_theta, t, side='right') - 1]
            N_t = (N0 if t < tau else N1)
            mu_t = max(theta_t * N_t, epsilon)
            ll_total += y[t] * np.log(mu_t) - mu_t
        
        return ll_total

    def optimize_N_params(tau, tau_theta, theta_values):
        """
        Optimise les paramètres N0 et N1 avec MLE.
        """
        alpha_post_N0 = alpha_prior + np.sum(y[:tau])
        beta_post_N0 = beta_prior + theta_values[0] * tau

        alpha_post_N1 = alpha_prior + np.sum(y[tau:])
        beta_post_N1 = beta_prior + theta_values[-1] * (T - tau)

        N0_est = max((alpha_post_N0 - 1) / beta_post_N0, 0) if alpha_post_N0 > 1 else 0
        N1_est = max((alpha_post_N1 - 1) / beta_post_N1, 0) if alpha_post_N1 > 1 else 0

        return N0_est, N1_est

    # Exploration de tous les scénarios possibles
    best_params = None
    log_likelihood_best = -np.inf
    AIC_best = np.inf

    # Tester tous les tau possibles
    for tau in range(1, T - 1):

        # Tester différentes partitions de θ en segments
        for k in range(1, max_theta_segments + 1):
            possible_tau_theta = list(itertools.combinations(range(1, T - 1), k - 1))

            for tau_theta in possible_tau_theta:
                tau_theta = [0] + list(tau_theta) + [T]

                # Optimiser θ pour chaque segment
                theta_values = []
                for i in range(len(tau_theta) - 1):
                    start, end = tau_theta[i], tau_theta[i + 1]
                    valid_data = y[start:end][~np.isnan(y[start:end])]
                    theta_values.append(max(np.mean(valid_data), epsilon) if len(valid_data) > 0 else epsilon)

                # Optimiser N0 et N1
                N0, N1 = optimize_N_params(tau, tau_theta, theta_values)

                # Calculer la log-vraisemblance
                log_likelihood = compute_log_likelihood(tau, tau_theta, theta_values)

                # Calculer l'AIC (Akaike Information Criterion)
                num_params = 2 + len(theta_values)  # 2 pour N0 et N1, + len(theta_values) pour les θ
                AIC = -2 * log_likelihood + 2 * num_params

                # Mettre à jour si le modèle est meilleur
                if AIC < AIC_best:
                    AIC_best = AIC
                    log_likelihood_best = log_likelihood
                    best_params = (tau, tau_theta, N0, N1, theta_values)

    return best_params + (log_likelihood_best, AIC_best)
