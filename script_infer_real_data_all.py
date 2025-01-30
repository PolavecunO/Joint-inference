# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:07:36 2025

@author: Pol
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from inference.infer_abundance_MLE import infer_abundance_MLE
from inference.infer_abundance_MLE_complex_theta import infer_abundance_MLE_complex_theta
from inference.infer_abundance_MCMC import infer_abundance_MCMC
from inference.infer_abundance_MLE_theta_tau import infer_abundance_MLE_theta_tau
from data_simulation_treatment.process_yearly_data import process_yearly_data, invert_time_data

# === Paramètres ===
data_folder = "data"  # Dossier contenant les sous-dossiers
method = "MLE"  # Choisir parmi : "MLE", "MCMC", "MLE_theta_tau", "MLE_complex_theta", "all"
theta_transition_type = "sigmoid"  # Choisir parmi : "constant", "linear", "exponential", "stepwise", "logarithmic", "sigmoid"

num_samples = 1000  # Nombre d'échantillons pour MCMC
epsilon = 1e-6
seuil_correction = 0.034  
alpha_prior_N = 6.25
beta_prior_N = 0.1
alpha_prior_theta = 2.0
beta_prior_theta = 1.0
theta_values = [0.01, 0.1]  
theta_change_years = [1800]  

if method not in ["MLE", "MCMC", "MLE_theta_tau", "MLE_complex_theta", "all"]:
    print("Erreur : méthode inconnue. Choisissez parmi 'MLE', 'MCMC', 'MLE_theta_tau', 'MLE_complex_theta', 'all'.")
    sys.exit()

if not os.path.isdir(data_folder):
    raise FileNotFoundError(f"Le dossier '{data_folder}' n'existe pas.")

global_year_change_count = {}
subgenre_year_change_counts = {}

for subgenre in os.listdir(data_folder):
    subgenre_path = os.path.join(data_folder, subgenre)

    if not os.path.isdir(subgenre_path):
        continue

    files = [f for f in os.listdir(subgenre_path) if f.endswith(".csv")]

    if not files:
        print(f"Aucun fichier CSV trouvé dans '{subgenre}'.")
        continue

    subgenre_year_change_count = {}
    for file_name in files:
        file_path = os.path.join(subgenre_path, file_name)

        try:
            years, observations = process_yearly_data(file_path)
            count_data = invert_time_data(observations)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file_name}: {e}")
            continue

        T = len(count_data)

        if T == 0:
            print(f"Aucune donnée valide trouvée dans le fichier {file_name}.")
            continue

        theta_change_times = [
            len(years) - 1 - np.where(years[::-1] == year)[0][0]
            for year in theta_change_years if year in years
        ]

        y = count_data
        results = {}

        try:
            if method in ["MLE", "all"]:
                results["MLE"] = infer_abundance_MLE(y, T, epsilon, seuil_correction, alpha_prior_N, beta_prior_N, theta_values, theta_change_times)
            if method in ["MLE_complex_theta", "all"]:
                results["MLE_complex_theta"] = infer_abundance_MLE_complex_theta(y, T, epsilon, seuil_correction, alpha_prior_N, beta_prior_N, theta_values, theta_change_times, theta_transition_type)
            if method in ["MLE_theta_tau", "all"]:
                results["MLE_theta_tau"] = infer_abundance_MLE_theta_tau(y, T, epsilon, seuil_correction, alpha_prior_N, beta_prior_N, max_theta_segments=len(theta_values))
            if method in ["MCMC", "all"]:
                trace, _ = infer_abundance_MCMC(y, T, num_samples, epsilon, alpha_prior_N, beta_prior_N, alpha_prior_theta, beta_prior_theta)
                tau_post = trace.posterior["tau"].values.flatten()
                results["MCMC"] = int(np.median(tau_post))

        except Exception as e:
            print(f"Erreur lors de l'inférence ({method}) pour {file_name}: {e}")
            continue

        tau_est_year = years[::-1][results["MLE"][0]] if "MLE" in results and 0 <= results["MLE"][0] < T else None
        if tau_est_year is not None:
            subgenre_year_change_count[tau_est_year] = subgenre_year_change_count.get(tau_est_year, 0) + 1
            global_year_change_count[tau_est_year] = global_year_change_count.get(tau_est_year, 0) + 1

    subgenre_year_change_counts[subgenre] = subgenre_year_change_count

# Affichage global
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

for subgenre, counts in subgenre_year_change_counts.items():
    sorted_years = sorted(counts.keys())
    sorted_counts = [counts[year] for year in sorted_years]
    axes[0].plot(sorted_years, sorted_counts, marker="o", label=subgenre)

axes[0].set_title("Changements détectés par sous-genre")
axes[0].legend()
axes[0].grid()

sorted_years = sorted(global_year_change_count.keys())
sorted_counts = [global_year_change_count[year] for year in sorted_years]
axes[1].plot(sorted_years, sorted_counts, marker="o", color="green", label="Global")

axes[1].set_title("Changements détectés (global)")
axes[1].legend()
axes[1].grid()

theta_years = list(range(min(theta_change_years), max(theta_change_years) + 1))
theta_plot_values = [theta_values[0] if year < theta_change_years[0] else theta_values[-1] for year in theta_years]
axes[2].plot(theta_years, theta_plot_values, marker="o", color="purple", label="Theta")

axes[2].set_title("Variations de theta")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()
