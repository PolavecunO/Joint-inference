# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:06:17 2025

@author: Pol
"""

import numpy as np

import numpy as np

def simulate_abundance_data(
    N0_real, N1_real, T, tau_real, 
    theta_values=[0.1], theta_change_times=None, theta_transition_type="constant",
    missing_data=False, p=0.1, Gamma=False, beta_gamma=0.1, random_seed=42
):
    """
    Simule des données de comptage de population avec transitions complexes pour \(\theta\).

    Arguments :
    - N0_real, N1_real : Tailles de population avant et après le changement de population.
    - T : Nombre total d'observations.
    - tau_real : Temps de changement des tailles de population.
    - theta_values : Liste des valeurs de \(\theta\) pour les périodes définies.
    - theta_change_times : Liste des temps de changement de \(\theta\) (indices, optionnel).
    - theta_transition_type : Type de transition entre les valeurs de \(\theta\) :
        - `"constant"` (default) : \(\theta\) change instantanément.
        - `"linear"` : Transition linéaire entre les valeurs de \(\theta\).
        - `"exponential"` : Transition exponentielle entre valeurs de \(\theta\).
        - `"logarithmic"` : Transition logarithmique.
        - `"sigmoid"` : Transition en sigmoïde.
    - missing_data : Si True, introduit des valeurs manquantes avec probabilité `p`.
    - Gamma : Si True, les tailles \(N_t\) sont tirées d'une distribution Gamma.
    - beta_gamma : Paramètre beta de la distribution Gamma.
    - random_seed : Graine aléatoire pour la reproductibilité.

    Retourne :
    - y : Observations simulées, avec éventuelles valeurs manquantes.
    """
    np.random.seed(random_seed)

    # Gestion des changements de theta
    if not theta_change_times or len(theta_values) == 1:
        theta_change_times = []  # Pas de changement
        theta_values = [theta_values[0]]

    # Étendre les temps de changement pour inclure les bornes de début et de fin
    theta_change_times = [0] + theta_change_times + [T]
    assert len(theta_values) == len(theta_change_times) - 1, \
        "Le nombre de valeurs dans `theta_values` doit correspondre aux intervalles définis par `theta_change_times`."

    def get_theta_at(t):
        """Retourne la valeur de \(\theta\) pour un instant \(t\) avec la transition spécifiée."""
        for i in range(len(theta_change_times) - 1):
            t_start, t_end = theta_change_times[i], theta_change_times[i + 1]
            theta_start, theta_end = theta_values[i], theta_values[i + 1] if i + 1 < len(theta_values) else theta_values[i]

            if t_start <= t < t_end:
                if theta_transition_type == "constant":
                    return theta_start
                elif theta_transition_type == "linear":
                    return theta_start + (theta_end - theta_start) * (t - t_start) / (t_end - t_start)
                elif theta_transition_type == "exponential":
                    return theta_start * ((theta_end / theta_start) ** ((t - t_start) / (t_end - t_start)))
                elif theta_transition_type == "logarithmic":
                    return theta_start + (theta_end - theta_start) * np.log1p(t - t_start) / np.log1p(t_end - t_start)
                elif theta_transition_type == "sigmoid":
                    x = (t - t_start) / (t_end - t_start) * 12 - 6  # Normalisation entre -6 et +6
                    return theta_start + (theta_end - theta_start) / (1 + np.exp(-x))
                else:
                    raise ValueError(f"Type de transition '{theta_transition_type}' non reconnu.")
        return theta_values[-1]  # Valeur par défaut pour la dernière période

    # Générer les tailles \(N_t\)
    if Gamma:
        N0_t = np.random.gamma(N0_real / beta_gamma, beta_gamma, size=min(T, tau_real))
        N1_t = np.random.gamma(N1_real / beta_gamma, beta_gamma, size=max(0, (T - tau_real)))
        N_t = np.concatenate([N0_t, N1_t])
    else:
        N_t = np.array([N0_real] * tau_real + [N1_real] * (T - tau_real))

    # Générer les observations \(y_t\) en utilisant les valeurs de \(\theta\)
    y = np.zeros(T)
    for t in range(T):
        theta_t = get_theta_at(t)
        y[t] = np.random.poisson(theta_t * N_t[t])

    # Ajouter des valeurs manquantes si demandé
    if missing_data:
        mask_missing = np.random.uniform(0, 1, size=T) < p
        y = np.where(mask_missing, np.nan, y)

    return y
