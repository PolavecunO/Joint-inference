# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:24:18 2025

@author: Pol
"""

from test_abundance_model import test_abundance_model
from visualization_functions import plot_error_heatmaps, plot_error_difference_heatmaps, plot_error_vs_param
import numpy as np

# Définir les plages des paramètres
N0_values = np.linspace(150, 1000, 10)
N1_values = np.linspace(150, 1000, 10)
param_values = [0.01, 0.1, 1.0]  # Exemple pour beta_gamma
binary_param = [False, True]  # Exemple pour Gamma
param_name = "beta_gamma"
binary_name = "Gamma"

# Définir les paramètres fixes
fixed_params = {
    "T": 100,
    "tau_real": 50,
    "theta": 0.1,
    "missing_data": True,
    "p": 0.1,
    "epsilon": 1e-6,
    "seuil_correction": 0.034,
    "alpha_prior": 6.25,
    "beta_prior": 0.01,
    "beta_gamma": 0.01
}

# Exécuter le test
results_matrix, meta_info = test_abundance_model(
    N0_values=N0_values,
    N1_values=N1_values,
    param_values=param_values,
    param_name=param_name,
    binary_param=binary_param,
    binary_name=binary_name,
    fixed_params=fixed_params
)

# Visualisation des résultats
plot_error_heatmaps(results_matrix, meta_info)
plot_error_difference_heatmaps(results_matrix, meta_info, binary_name, param_name, num_per_window=10)
plot_error_vs_param(results_matrix, meta_info, param_x=param_name, param_group=binary_name, averaging_params=["N0", "N1"])
